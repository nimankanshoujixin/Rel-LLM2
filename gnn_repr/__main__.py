from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything

from gnn_repr.builder import GNNRepresentationPretrainer
from main import load_or_create_stypes
from precompute.graph_cache import build_text_embedder_cfg, cache_paths, maybe_mask_autocomplete_columns, resolve_db
from relbench.datasets import get_dataset
from relbench.modeling.graph import make_pkey_fkey_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a label-free GNN representation artifact.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/.cache/relbench_examples"))
    parser.add_argument("--text_embedder", type=str, default="mpnet", choices=["glove", "mpnet"])
    parser.add_argument("--text_embedder_path", type=str, default="./cache")
    parser.add_argument("--embed_batch_size", type=int, default=256)
    parser.add_argument("--embed_devices", type=str, default="")
    parser.add_argument("--embed_chunk_size", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="artifacts/gnn_repr")
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=64)
    parser.add_argument("--aggr", type=str, default="mean")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_steps", type=int, default=1024)
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--edge_loss_weight", type=float, default=1.0)
    parser.add_argument("--recon_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_samples_per_relation", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--require_cache", action="store_true")
    return parser.parse_args()


def build_graph(args: argparse.Namespace, device: torch.device):
    dataset = get_dataset(args.dataset, download=args.download)
    cache_profile = "gnn_repr_train_visible"
    db = resolve_db(dataset, cache_profile)
    maybe_mask_autocomplete_columns(db, args.dataset, cache_profile)
    cache_root = Path(os.path.expanduser(args.cache_dir))
    stypes_cache_path, materialized_cache_dir = cache_paths(cache_root, args.dataset, cache_profile)
    col_to_stype_dict = load_or_create_stypes(db, stypes_cache_path, rank=0)
    expected_cache_files = [materialized_cache_dir / f"{table_name}.pt" for table_name in sorted(db.table_dict)]
    if args.require_cache and not all(path.exists() for path in expected_cache_files):
        missing = [str(path.name) for path in expected_cache_files if not path.exists()]
        raise FileNotFoundError(
            "Missing GNN representation cache files. "
            f"Expected profile '{cache_profile}' under {materialized_cache_dir}. Missing tables: {missing}"
        )
    text_embedder_cfg = build_text_embedder_cfg(
        args=args,
        table_names=sorted(db.table_dict),
        cache_dir=materialized_cache_dir,
    )
    data, col_stats_dict = make_pkey_fkey_graph(
        db,
        col_to_stype_dict=col_to_stype_dict,
        text_embedder_cfg=text_embedder_cfg,
        cache_dir=str(materialized_cache_dir),
    )
    return data, col_stats_dict


def build_node_type_loaders(args: argparse.Namespace, data) -> dict[str, NeighborLoader]:
    loaders: dict[str, NeighborLoader] = {}
    num_neighbors = [max(1, int(args.num_neighbors / (2 ** i))) for i in range(args.num_layers)]
    for node_type in data.node_types:
        num_nodes = int(data[node_type].num_nodes)
        if num_nodes <= 1:
            continue
        loaders[node_type] = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            input_nodes=(node_type, torch.arange(num_nodes)),
            batch_size=min(args.batch_size, num_nodes),
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
    if not loaders:
        raise RuntimeError("No node types available for GNN representation training.")
    return loaders


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data, col_stats_dict = build_graph(args, device)
    loaders = build_node_type_loaders(args, data)
    trainer = GNNRepresentationPretrainer(
        data,
        col_stats_dict,
        channels=args.channels,
        num_layers=args.num_layers,
        aggr=args.aggr,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        edge_loss_weight=args.edge_loss_weight,
        recon_loss_weight=args.recon_loss_weight,
        edge_samples_per_relation=args.edge_samples_per_relation,
    ).to(device)
    optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr, weight_decay=args.wd)

    iterators = {node_type: iter(loader) for node_type, loader in loaders.items()}
    node_types = sorted(loaders)
    for step in range(1, args.train_steps + 1):
        node_type = node_types[(step - 1) % len(node_types)]
        try:
            batch = next(iterators[node_type])
        except StopIteration:
            iterators[node_type] = iter(loaders[node_type])
            batch = next(iterators[node_type])
        batch = batch.to(device)
        trainer.train()
        optimizer.zero_grad(set_to_none=True)
        outputs = trainer(batch, node_type)
        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_interval == 0 or step == args.train_steps:
            print(
                f"[GNN-REPR] step={step}/{args.train_steps} "
                f"loss={float(outputs['loss'].detach().cpu()):.6f} "
                f"recon={float(outputs['recon_loss'].cpu()):.6f} "
                f"edge={float(outputs['edge_loss'].cpu()):.6f} "
                f"root={node_type}"
            )

    artifact = trainer.build_artifact(
        dataset=args.dataset,
        channels=args.channels,
        num_layers=args.num_layers,
        aggr=args.aggr,
        train_steps=args.train_steps,
        graph_cache_profile="gnn_repr_train_visible",
        downstream_graph_cache_profile="gnn_repr_full_horizon",
    )
    output_path = Path(args.output_root) / args.dataset / "gnn_repr.pt"
    saved_path = trainer.save_artifact(artifact, output_path)
    print(f"Saved GNN representation artifact to {saved_path}")


if __name__ == "__main__":
    main()
