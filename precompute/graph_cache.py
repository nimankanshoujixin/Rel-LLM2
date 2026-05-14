from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_frame
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from torch_frame.data import Dataset
from torch_frame.data.dataset import DataFrameToTensorFrameConverter
from torch_frame.data.stats import StatType, compute_col_stats

from main import get_autocomplete_task_names, load_or_create_stypes
from relbench.datasets import get_dataset
from relbench.modeling.utils import remove_pkey_fkey
from relbench.tasks import get_task
from text_embedder import TextEmbedding
from utils import is_autocomplete_task


CACHE_PROFILES = {
    "task_default": {
        "stypes_name": "stypes.json",
        "materialized_name": "materialized",
        "mask_autocomplete": False,
    },
    "autocomplete_masked": {
        "stypes_name": "stypes_autocomplete_masked.json",
        "materialized_name": "materialized_autocomplete_masked",
        "mask_autocomplete": True,
    },
    "gnn_repr_train_visible": {
        "stypes_name": "stypes_gnn_repr_train_visible.json",
        "materialized_name": "materialized_gnn_repr_train_visible",
        "mask_autocomplete": True,
    },
    "gnn_repr_full_horizon": {
        "stypes_name": "stypes_gnn_repr_train_visible.json",
        "materialized_name": "materialized_gnn_repr_full_horizon",
        "mask_autocomplete": True,
        "col_stats_source_profile": "gnn_repr_train_visible",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute graph tensor-frame caches.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--cache-profile",
        type=str,
        default="gnn_repr_train_visible",
        choices=sorted(CACHE_PROFILES),
    )
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"))
    parser.add_argument("--text_embedder", type=str, default="mpnet", choices=["glove", "mpnet"])
    parser.add_argument("--text_embedder_path", type=str, default="./cache")
    parser.add_argument("--embed_batch_size", type=int, default=256)
    parser.add_argument("--embed_devices", type=str, default="")
    parser.add_argument("--embed_chunk_size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--force_cpu_aggregate",
        action="store_true",
        help="Force text embeddings to be materialized back on CPU before torch_frame aggregation.",
    )
    parser.add_argument("--row_shard_count", type=int, default=1)
    parser.add_argument("--row_shard_index", type=int, default=0)
    parser.add_argument("--shard_output_root", type=str, default="")
    parser.add_argument("--merge_shards_only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--copy_from_equivalent", action="store_true")
    parser.add_argument("--table_names", type=str, default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def cache_paths(cache_root: Path, dataset_name: str, cache_profile: str) -> tuple[Path, Path]:
    profile = CACHE_PROFILES[cache_profile]
    dataset_root = cache_root / dataset_name
    return dataset_root / profile["stypes_name"], dataset_root / profile["materialized_name"]


def resolve_db(dataset, cache_profile: str):
    if cache_profile == "task_default":
        return dataset.get_db(upto_test_timestamp=True)
    if cache_profile == "autocomplete_masked":
        return dataset.get_db(upto_test_timestamp=False)
    if cache_profile == "gnn_repr_train_visible":
        return dataset.get_db(upto_test_timestamp=False).upto(dataset.val_timestamp)
    if cache_profile == "gnn_repr_full_horizon":
        return dataset.get_db(upto_test_timestamp=False)
    raise ValueError(f"Unsupported cache profile: {cache_profile}")


def autocomplete_target_columns(dataset_name: str) -> dict[str, set[str]]:
    masked_cols_by_table: dict[str, set[str]] = {}
    for task_name in get_autocomplete_task_names(dataset_name):
        task = get_task(dataset_name, task_name, download=False)
        if not is_autocomplete_task(task):
            continue
        masked_cols_by_table.setdefault(task.entity_table, set()).add(task.target_col)
    return masked_cols_by_table


def maybe_mask_autocomplete_columns(db, dataset_name: str, cache_profile: str) -> dict[str, set[str]]:
    if not CACHE_PROFILES[cache_profile].get("mask_autocomplete", False):
        return {}
    masked_cols_by_table = autocomplete_target_columns(dataset_name)
    for table_name, cols_to_drop in masked_cols_by_table.items():
        if table_name not in db.table_dict:
            continue
        table = db.table_dict[table_name]
        drop_cols = [col for col in cols_to_drop if col in table.df.columns]
        if drop_cols:
            table.df = table.df.drop(columns=drop_cols)
    return masked_cols_by_table


def build_text_embedder_cfg(
    *,
    args: argparse.Namespace,
    table_names: list[str],
    cache_dir: Path,
) -> TextEmbedderConfig:
    all_ready = all((cache_dir / f"{table_name}.pt").exists() for table_name in table_names)
    if all_ready:
        return TextEmbedderConfig(
            text_embedder=lambda rows: torch.empty((len(rows), 0)),
            batch_size=args.embed_batch_size,
        )

    embed_devices = split_csv(args.embed_devices)
    embed_device = args.device
    if embed_device is None:
        embed_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    text_embedder = TextEmbedding(
        args.text_embedder,
        args.text_embedder_path,
        device=embed_device,
        devices=embed_devices,
        encode_batch_size=args.embed_batch_size,
        multi_process_chunk_size=(args.embed_chunk_size or None),
        show_progress_bar=False,
    )
    return TextEmbedderConfig(
        text_embedder=text_embedder,
        batch_size=args.embed_batch_size,
    )


def materialize_table(
    *,
    table_name: str,
    table,
    col_to_stype_dict: dict[str, dict[str, stype]],
    text_embedder_cfg: TextEmbedderConfig,
    output_dir: Path,
    skip_existing: bool,
    col_stats_override: dict[str, dict[StatType, Any]] | None = None,
) -> None:
    path = output_dir / f"{table_name}.pt"
    if skip_existing and path.exists():
        print(f"[GRAPH-CACHE] skip existing table={table_name} path={path}")
        return

    start_time = time.perf_counter()
    df = table.df
    col_to_stype = dict(col_to_stype_dict[table_name])
    remove_pkey_fkey(col_to_stype, table)

    if len(col_to_stype) == 0:
        col_to_stype = {"__const__": stype.numerical}
        fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
        df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})

    if col_stats_override is None:
        dataset = Dataset(
            df=df,
            col_to_stype=col_to_stype,
            col_to_text_embedder_cfg=text_embedder_cfg,
        ).materialize(path=str(path))
        del dataset
    else:
        pattern_maps = build_pattern_maps(
            df=df,
            col_to_stype=col_to_stype,
            text_embedder_cfg=text_embedder_cfg,
        )
        converter = DataFrameToTensorFrameConverter(
            col_to_stype=col_to_stype,
            col_stats=copy.deepcopy(col_stats_override),
            target_col=None,
            col_to_sep=pattern_maps["col_to_sep"],
            col_to_text_embedder_cfg=pattern_maps["col_to_text_embedder_cfg"],
            col_to_time_format=pattern_maps["col_to_time_format"],
        )
        tensor_frame = converter(df, device=None)
        frozen_col_stats = update_embedding_col_stats(
            tensor_frame=tensor_frame,
            col_stats=copy.deepcopy(col_stats_override),
        )
        torch_frame.save(tensor_frame, frozen_col_stats, str(path))
    elapsed = time.perf_counter() - start_time
    row_count = len(df)
    rows_per_sec = (row_count / elapsed) if elapsed > 0 else float("inf")
    print(
        f"[GRAPH-CACHE] materialized table={table_name} path={path} "
        f"rows={row_count} elapsed_sec={elapsed:.2f} rows_per_sec={rows_per_sec:.2f}"
    )


def prepare_table_dataframe(
    *,
    table,
    col_to_stype_dict: dict[str, dict[str, stype]],
    table_name: str,
) -> tuple[pd.DataFrame, dict[str, stype]]:
    df = table.df
    col_to_stype = dict(col_to_stype_dict[table_name])
    remove_pkey_fkey(col_to_stype, table)

    if len(col_to_stype) == 0:
        col_to_stype = {"__const__": stype.numerical}
        fkey_dict = {key: df[key] for key in table.fkey_col_to_pkey_table}
        df = pd.DataFrame({"__const__": np.ones(len(table.df)), **fkey_dict})
    return df, col_to_stype


def compute_global_col_stats(
    *,
    df: pd.DataFrame,
    col_to_stype: dict[str, stype],
) -> dict[str, dict[StatType, Any]]:
    col_stats: dict[str, dict[StatType, Any]] = {}
    for col, col_stype in col_to_stype.items():
        col_stats[col] = compute_col_stats(df[col], col_stype)
    return col_stats


def build_pattern_maps(
    *,
    df: pd.DataFrame,
    col_to_stype: dict[str, stype],
    text_embedder_cfg: TextEmbedderConfig,
) -> dict[str, dict[str, Any]]:
    text_embedder_map = {
        col: text_embedder_cfg
        for col, col_stype in col_to_stype.items()
        if col_stype == stype.text_embedded
    }
    sep_map = {
        col: None
        for col, col_stype in col_to_stype.items()
        if col_stype == stype.multicategorical
    }
    time_format_map = {
        col: None
        for col, col_stype in col_to_stype.items()
        if col_stype == stype.timestamp
    }
    return {
        "col_to_text_embedder_cfg": text_embedder_map,
        "col_to_sep": sep_map,
        "col_to_time_format": time_format_map,
    }


def update_embedding_col_stats(
    *,
    tensor_frame,
    col_stats: dict[str, dict[StatType, Any]],
) -> dict[str, dict[StatType, Any]]:
    updated = copy.deepcopy(col_stats)
    if torch_frame.embedding in tensor_frame.feat_dict:
        offset = tensor_frame.feat_dict[torch_frame.embedding].offset
        emb_dim_list = offset[1:] - offset[:-1]
        for i, col_name in enumerate(tensor_frame.col_names_dict[torch_frame.embedding]):
            updated[col_name][StatType.EMB_DIM] = int(emb_dim_list[i])
    return updated


def row_shard_bounds(num_rows: int, row_shard_count: int, row_shard_index: int) -> tuple[int, int]:
    if row_shard_count <= 0:
        raise ValueError("row_shard_count must be positive.")
    if row_shard_index < 0 or row_shard_index >= row_shard_count:
        raise ValueError(
            f"row_shard_index must be in [0, {row_shard_count}), got {row_shard_index}."
        )
    boundaries = np.linspace(0, num_rows, row_shard_count + 1, dtype=int)
    return int(boundaries[row_shard_index]), int(boundaries[row_shard_index + 1])


def shard_cache_path(shard_output_root: Path, table_name: str, shard_index: int) -> Path:
    return shard_output_root / f"{table_name}.part{shard_index:03d}.pt"


def materialize_table_row_shard(
    *,
    table_name: str,
    table,
    col_to_stype_dict: dict[str, dict[str, stype]],
    text_embedder_cfg: TextEmbedderConfig,
    shard_output_root: Path,
    row_shard_count: int,
    row_shard_index: int,
    col_stats_override: dict[str, dict[StatType, Any]] | None = None,
) -> Path:
    start_time = time.perf_counter()
    full_df, col_to_stype = prepare_table_dataframe(
        table=table,
        col_to_stype_dict=col_to_stype_dict,
        table_name=table_name,
    )
    start, end = row_shard_bounds(len(full_df), row_shard_count, row_shard_index)
    shard_df = full_df.iloc[start:end].reset_index(drop=True)
    shard_path = shard_cache_path(shard_output_root, table_name, row_shard_index)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    global_col_stats = (
        copy.deepcopy(col_stats_override)
        if col_stats_override is not None
        else compute_global_col_stats(df=full_df, col_to_stype=col_to_stype)
    )
    pattern_maps = build_pattern_maps(
        df=full_df,
        col_to_stype=col_to_stype,
        text_embedder_cfg=text_embedder_cfg,
    )
    converter = DataFrameToTensorFrameConverter(
        col_to_stype=col_to_stype,
        col_stats=global_col_stats,
        target_col=None,
        col_to_sep=pattern_maps["col_to_sep"],
        col_to_text_embedder_cfg=pattern_maps["col_to_text_embedder_cfg"],
        col_to_time_format=pattern_maps["col_to_time_format"],
    )
    tensor_frame = converter(shard_df, device=None)
    shard_col_stats = update_embedding_col_stats(tensor_frame=tensor_frame, col_stats=global_col_stats)
    torch_frame.save(tensor_frame, shard_col_stats, str(shard_path))
    elapsed = time.perf_counter() - start_time
    row_count = len(shard_df)
    rows_per_sec = (row_count / elapsed) if elapsed > 0 else float("inf")
    print(
        f"[GRAPH-CACHE] materialized row shard table={table_name} "
        f"rows={start}:{end} shard={row_shard_index}/{row_shard_count} path={shard_path} "
        f"elapsed_sec={elapsed:.2f} shard_rows={row_count} rows_per_sec={rows_per_sec:.2f}"
    )
    return shard_path


def merge_row_shards(
    *,
    table_name: str,
    row_shard_count: int,
    shard_output_root: Path,
    output_dir: Path,
    skip_existing: bool,
) -> Path:
    final_path = output_dir / f"{table_name}.pt"
    if skip_existing and final_path.exists():
        print(f"[GRAPH-CACHE] skip existing merged table={table_name} path={final_path}")
        return final_path

    start_time = time.perf_counter()
    tensor_frames = []
    col_stats = None
    for shard_index in range(row_shard_count):
        shard_path = shard_cache_path(shard_output_root, table_name, shard_index)
        if not shard_path.exists():
            raise FileNotFoundError(f"Expected shard cache at {shard_path}")
        tensor_frame, shard_col_stats = torch_frame.load(str(shard_path), device=None)
        tensor_frames.append(tensor_frame)
        if col_stats is None:
            col_stats = shard_col_stats
    merged = torch_frame.cat(tensor_frames, dim=0)
    torch_frame.save(merged, col_stats, str(final_path))
    elapsed = time.perf_counter() - start_time
    row_count = len(merged)
    rows_per_sec = (row_count / elapsed) if elapsed > 0 else float("inf")
    print(
        f"[GRAPH-CACHE] merged row shards table={table_name} path={final_path} "
        f"rows={row_count} elapsed_sec={elapsed:.2f} rows_per_sec={rows_per_sec:.2f}"
    )
    return final_path


def select_tables(
    db,
    *,
    requested_tables: list[str],
    num_shards: int,
    shard_index: int,
) -> list[str]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}.")

    table_names = sorted(db.table_dict)
    if requested_tables:
        requested = set(requested_tables)
        table_names = [table_name for table_name in table_names if table_name in requested]
    return [table_name for index, table_name in enumerate(table_names) if index % num_shards == shard_index]


def equivalent_profile(cache_profile: str) -> str | None:
    if cache_profile == "autocomplete_masked":
        return "task_default"
    return None


def col_stats_source_profile(cache_profile: str) -> str | None:
    return CACHE_PROFILES[cache_profile].get("col_stats_source_profile")


def load_cached_col_stats(
    *,
    cache_root: Path,
    dataset_name: str,
    cache_profile: str,
    table_name: str,
) -> dict[str, dict[StatType, Any]]:
    _, materialized_dir = cache_paths(cache_root, dataset_name, cache_profile)
    cache_path = materialized_dir / f"{table_name}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Expected source col_stats cache at {cache_path} for table={table_name} "
            f"while building profile={cache_profile}."
        )
    _, col_stats = torch_frame.load(str(cache_path), device=None)
    if col_stats is None:
        raise RuntimeError(f"Cached tensor frame at {cache_path} does not contain col_stats.")
    return col_stats


def clone_equivalent_cache_if_possible(
    *,
    cache_root: Path,
    dataset_name: str,
    cache_profile: str,
) -> bool:
    src_profile = equivalent_profile(cache_profile)
    if src_profile is None:
        return False
    masked_cols = autocomplete_target_columns(dataset_name)
    if masked_cols:
        return False
    src_stypes, src_materialized = cache_paths(cache_root, dataset_name, src_profile)
    dst_stypes, dst_materialized = cache_paths(cache_root, dataset_name, cache_profile)
    if not src_stypes.exists() or not src_materialized.exists():
        return False
    dst_stypes.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_stypes, dst_stypes)
    if dst_materialized.exists():
        shutil.rmtree(dst_materialized)
    shutil.copytree(src_materialized, dst_materialized)
    print(
        f"[GRAPH-CACHE] cloned equivalent cache profile={cache_profile} "
        f"from={src_profile} dataset={dataset_name}"
    )
    return True


def build_graph_cache(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    dataset = get_dataset(args.dataset, download=args.download)
    db = resolve_db(dataset, args.cache_profile)
    masked_cols = maybe_mask_autocomplete_columns(db, args.dataset, args.cache_profile)
    cache_root = Path(os.path.expanduser(args.cache_dir))
    stypes_cache_path, materialized_cache_dir = cache_paths(cache_root, args.dataset, args.cache_profile)
    materialized_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.copy_from_equivalent and clone_equivalent_cache_if_possible(
        cache_root=cache_root,
        dataset_name=args.dataset,
        cache_profile=args.cache_profile,
    ):
        return {
            "dataset": args.dataset,
            "cache_profile": args.cache_profile,
            "mode": "cloned_equivalent",
            "masked_tables": {key: sorted(value) for key, value in masked_cols.items()},
        }

    col_to_stype_dict = load_or_create_stypes(db, stypes_cache_path, rank=0)
    selected_tables = select_tables(
        db,
        requested_tables=split_csv(args.table_names),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    source_profile = col_stats_source_profile(args.cache_profile)
    text_embedder_cfg = build_text_embedder_cfg(
        args=args,
        table_names=selected_tables,
        cache_dir=materialized_cache_dir,
    )
    if args.force_cpu_aggregate:
        print("[GRAPH-CACHE] force_cpu_aggregate enabled; text embeddings must return to CPU before materialization")
    text_embedder = getattr(text_embedder_cfg, "text_embedder", None)
    try:
        for table_name in selected_tables:
            frozen_col_stats = None
            if source_profile is not None:
                frozen_col_stats = load_cached_col_stats(
                    cache_root=cache_root,
                    dataset_name=args.dataset,
                    cache_profile=source_profile,
                    table_name=table_name,
                )
            if args.merge_shards_only:
                if not args.shard_output_root:
                    raise ValueError("--merge_shards_only requires --shard_output_root.")
                merge_row_shards(
                    table_name=table_name,
                    row_shard_count=args.row_shard_count,
                    shard_output_root=Path(args.shard_output_root),
                    output_dir=materialized_cache_dir,
                    skip_existing=args.skip_existing,
                )
            elif args.row_shard_count > 1:
                if not args.shard_output_root:
                    raise ValueError("row sharding requires --shard_output_root.")
                materialize_table_row_shard(
                    table_name=table_name,
                    table=db.table_dict[table_name],
                    col_to_stype_dict=col_to_stype_dict,
                    text_embedder_cfg=text_embedder_cfg,
                    shard_output_root=Path(args.shard_output_root),
                    row_shard_count=args.row_shard_count,
                    row_shard_index=args.row_shard_index,
                    col_stats_override=frozen_col_stats,
                )
            else:
                materialize_table(
                    table_name=table_name,
                    table=db.table_dict[table_name],
                    col_to_stype_dict=col_to_stype_dict,
                    text_embedder_cfg=text_embedder_cfg,
                    output_dir=materialized_cache_dir,
                    skip_existing=args.skip_existing,
                    col_stats_override=frozen_col_stats,
                )
    finally:
        close_fn = getattr(text_embedder, "close", None)
        if callable(close_fn):
            close_fn()

    total_elapsed = time.perf_counter() - total_start

    return {
        "dataset": args.dataset,
        "cache_profile": args.cache_profile,
        "mode": "materialized",
        "tables": selected_tables,
        "masked_tables": {key: sorted(value) for key, value in masked_cols.items()},
        "output_dir": str(materialized_cache_dir),
        "elapsed_sec": total_elapsed,
    }


def main() -> None:
    args = parse_args()
    result = build_graph_cache(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
