import argparse
import copy
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pooch
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from tqdm import tqdm

from model import Model
from relbench.base import Dataset, TaskType
from relbench.datasets import get_dataset
from relbench.modeling.graph import (
    AttachTargetTransform,
    NodeTrainTableInput,
    get_node_train_table_input,
    make_pkey_fkey_graph,
)
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task
from text_embedder import TextEmbedding
from torch_frame import stype
from torch_frame.config.text_embedder import TextEmbedderConfig
from utils import task_info

warnings.filterwarnings(
    "ignore",
    message="cuDNN SDPA backward got grad_output.strides() != output.strides()",
)
warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["CURL_CA_BUNDLE"] = ""  # huggingface connection issue
os.environ.setdefault("NCCL_P2P_DISABLE", "1")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def load_or_create_stypes(db, cache_path: Path, rank: int) -> Dict[str, Dict[str, stype]]:
    try:
        with open(cache_path, "r") as f:
            col_to_stype_dict = json.load(f)
        for table, col_to_stype in col_to_stype_dict.items():
            for col, stype_str in col_to_stype.items():
                col_to_stype[col] = stype(stype_str)
        return col_to_stype_dict
    except FileNotFoundError:
        if is_distributed() and rank != 0:
            raise RuntimeError(
                f"Expected stype cache to exist at {cache_path} after rank 0 preparation."
            )
        col_to_stype_dict = get_stype_proposal(db)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(col_to_stype_dict, f, indent=2, default=str)
        return col_to_stype_dict


def graph_cache_ready(db, cache_dir: Path) -> bool:
    if not cache_dir.exists():
        return False
    return all((cache_dir / f"{table_name}.pt").exists() for table_name in db.table_dict)


def relbench_db_cache_ready(dataset_name: str) -> bool:
    db_path = Path(pooch.os_cache("relbench")) / dataset_name / "db"
    return db_path.exists() and any(db_path.iterdir())


def relbench_task_cache_ready(dataset_name: str, task_name: str) -> bool:
    task_path = Path(pooch.os_cache("relbench")) / dataset_name / "tasks" / task_name
    return task_path.exists() and all((task_path / f"{split}.parquet").exists() for split in ("train", "val", "test"))


def relbench_artifacts_ready(dataset_name: str, task_name: str) -> bool:
    return relbench_db_cache_ready(dataset_name) and relbench_task_cache_ready(dataset_name, task_name)


def load_dataset_task_and_db(
    args: argparse.Namespace,
    rank: int,
) -> tuple[Dataset, Any, Any]:
    if is_distributed() and not relbench_artifacts_ready(args.dataset, args.task):
        if rank == 0:
            get_dataset(args.dataset, download=True)
            get_task(args.dataset, args.task, download=True)
        barrier()

    dataset: Dataset = get_dataset(args.dataset, download=False)
    db = dataset.get_db()
    task = get_task(args.dataset, args.task, download=False)
    task.name = args.task
    return dataset, db, task


def build_text_embedder_cfg(
    args: argparse.Namespace,
    db,
    cache_dir: Path,
    device: torch.device,
) -> TextEmbedderConfig:
    if graph_cache_ready(db, cache_dir):
        return TextEmbedderConfig(
            text_embedder=lambda rows: torch.empty((len(rows), 0)),
            batch_size=256,
        )

    os.makedirs(args.text_embedder_path, exist_ok=True)
    text_embedder = TextEmbedding(args.text_embedder, args.text_embedder_path, device=device)
    return TextEmbedderConfig(text_embedder=text_embedder, batch_size=256)


def shard_indices(num_items: int, rank: int, world_size: int, pad_to_equal: bool) -> torch.Tensor:
    if world_size == 1:
        return torch.arange(num_items, dtype=torch.long)
    if num_items == 0:
        return torch.empty(0, dtype=torch.long)
    if pad_to_equal:
        shard_size = math.ceil(num_items / world_size)
        start = rank * shard_size
        end = start + shard_size
        return torch.arange(start, end, dtype=torch.long) % num_items
    return torch.tensor_split(torch.arange(num_items, dtype=torch.long), world_size)[rank]


def shard_optional_tensor(tensor: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[indices].clone()


def shard_node_train_input(
    table_input: NodeTrainTableInput,
    rank: int,
    world_size: int,
    pad_to_equal: bool,
) -> NodeTrainTableInput:
    indices = shard_indices(len(table_input.nodes[1]), rank, world_size, pad_to_equal)
    sharded_nodes = table_input.nodes[1][indices].clone()
    sharded_time = shard_optional_tensor(table_input.time, indices)
    sharded_target = shard_optional_tensor(table_input.target, indices)
    transform = None
    if sharded_target is not None:
        transform = AttachTargetTransform(table_input.nodes[0], sharded_target)
    return NodeTrainTableInput(
        nodes=(table_input.nodes[0], sharded_nodes),
        time=sharded_time,
        target=sharded_target,
        transform=transform,
    )


def build_loader_dict(
    args: argparse.Namespace,
    data,
    task,
    rank: int,
    world_size: int,
) -> tuple[Dict[str, NeighborLoader], str]:
    loader_dict: Dict[str, NeighborLoader] = {}
    entity_table = None
    for split in ["train", "val", "test"]:
        table = task.get_table(split, mask_input_cols=False)
        table_input = get_node_train_table_input(table=table, task=task)
        entity_table = table_input.nodes[0]
        if world_size > 1:
            table_input = shard_node_train_input(
                table_input,
                rank=rank,
                world_size=world_size,
                pad_to_equal=(split == "train"),
            )
        bs = args.batch_size if (split == "train" or args.val_size is None) else args.val_size
        if len(table_input.nodes[1]) == 0:
            raise RuntimeError(
                f"Rank {rank} received no input nodes for split '{split}'. "
                "Use fewer GPUs or a larger task split."
            )
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[int(args.num_neighbors / 2 ** i) for i in range(args.num_layers)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=bs,
            temporal_strategy=args.temporal_strategy,
            shuffle=split == "train",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=True,
        )
    assert entity_table is not None
    return loader_dict, entity_table


def gather_predictions(local_pred: np.ndarray) -> np.ndarray:
    if not is_distributed():
        return local_pred
    gathered: list[np.ndarray | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_pred)
    non_empty = [pred for pred in gathered if pred is not None and len(pred) > 0]
    if not non_empty:
        return np.array([])
    return np.concatenate(non_empty, axis=0)


@torch.no_grad()
def run_inference(
    loader: NeighborLoader,
    model: torch.nn.Module,
    task,
    args: argparse.Namespace,
    device: torch.device,
    demo_info=None,
    max_steps: int | None = None,
    progress_desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    pred_list = []
    target_list = []
    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    for step_idx, test_batch in enumerate(
        tqdm(
            loader,
            total=total_steps,
            desc=progress_desc,
            disable=not is_main_process(),
        ),
        start=1,
    ):
        if max_steps is not None and step_idx > max_steps:
            break
        test_batch = test_batch.to(device)
        pred = model(test_batch, task.entity_table, demo_info=demo_info, inference=True)
        if task.task_type == TaskType.REGRESSION:
            clamp_min, clamp_max = args.clamp_range
            assert clamp_min is not None and clamp_max is not None
            pred = torch.clamp(pred, clamp_min, clamp_max)

        if (args.model_type == "gnn" or args.output_mlp) and task.task_type in [
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTILABEL_CLASSIFICATION,
        ]:
            pred = torch.sigmoid(pred)

        pred = pred.view(-1) if len(pred.size()) > 1 and pred.size(1) == 1 else pred
        pred_list.append(pred.detach().cpu())
        if hasattr(test_batch[task.entity_table], "y"):
            target = test_batch[task.entity_table].y
            target = target.view(-1) if len(target.size()) > 1 and target.size(1) == 1 else target
            target_list.append(target.detach().cpu())

    if pred_list:
        local_pred = torch.cat(pred_list, dim=0).numpy()
    else:
        local_pred = np.array([])
    local_target = None
    if target_list:
        local_target = torch.cat(target_list, dim=0).numpy()
    return gather_predictions(local_pred), (
        gather_predictions(local_target) if local_target is not None else None
    )


@torch.no_grad()
def test(
    loader: NeighborLoader,
    model: torch.nn.Module,
    task,
    args: argparse.Namespace,
    device: torch.device,
    demo_info=None,
    max_steps: int | None = None,
    progress_desc: str | None = None,
) -> np.ndarray:
    pred, _ = run_inference(
        loader,
        model,
        task,
        args,
        device,
        demo_info=demo_info,
        max_steps=max_steps,
        progress_desc=progress_desc,
    )
    return pred


def evaluate_loader(
    loader: NeighborLoader,
    model: torch.nn.Module,
    task,
    args: argparse.Namespace,
    device: torch.device,
    demo_info=None,
    max_steps: int | None = None,
    progress_desc: str | None = None,
) -> dict[str, float]:
    pred, target = run_inference(
        loader,
        model,
        task,
        args,
        device,
        demo_info=demo_info,
        max_steps=max_steps,
        progress_desc=progress_desc,
    )
    if target is None:
        raise ValueError("Evaluation loader did not provide target labels.")
    if len(pred) != len(target):
        raise ValueError(
            f"The length of pred and target must be the same (got {len(pred)} and {len(target)})."
        )
    return {fn.__name__: fn(target, pred) for fn in task.metrics}


def resolve_max_steps(max_steps: int) -> int | None:
    return None if max_steps <= 0 else max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rel-stack")
    parser.add_argument("--task", type=str, default="user-engagement")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/relbench_examples"),
    )
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--aggr", type=str, default="sum")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neighbors", type=int, default=128)
    parser.add_argument(
        "--temporal_strategy",
        type=str,
        default="uniform",
        choices=["uniform", "last"],
    )
    parser.add_argument("--text_embedder", type=str, default="glove", choices=["glove", "mpnet"])
    parser.add_argument("--text_embedder_path", type=str, default="./cache")

    parser.add_argument(
        "--model_type",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model id, local model directory, or 'gnn'.",
    )
    parser.add_argument("--llm_frozen", action="store_true")
    parser.add_argument("--output_mlp", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_demo", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--loss_class_weight", nargs="+", type=float, default=None)

    parser.add_argument("--train_steps", type=int, default=2**15)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--pretrain_epochs", type=int, default=200)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=2**10)
    parser.add_argument("--test_steps", type=int, default=2**12)
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = setup_distributed()

    try:
        seed_everything(args.seed + rank)
        if torch.cuda.is_available():
            torch.set_num_threads(1)

        dataset, db, task = load_dataset_task_and_db(args, rank)

        stypes_cache_path = Path(f"{args.cache_dir}/{args.dataset}/stypes.json")
        materialized_cache_dir = Path(f"{args.cache_dir}/{args.dataset}/materialized")
        graph_artifacts_ready = stypes_cache_path.exists() and graph_cache_ready(db, materialized_cache_dir)
        if is_distributed() and not graph_artifacts_ready:
            if rank == 0:
                col_to_stype_dict = load_or_create_stypes(db, stypes_cache_path, rank)
                text_embedder_cfg = build_text_embedder_cfg(args, db, materialized_cache_dir, device)
                data, col_stats_dict = make_pkey_fkey_graph(
                    db,
                    col_to_stype_dict=col_to_stype_dict,
                    text_embedder_cfg=text_embedder_cfg,
                    cache_dir=str(materialized_cache_dir),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            barrier()
            if rank != 0:
                col_to_stype_dict = load_or_create_stypes(db, stypes_cache_path, rank)
                text_embedder_cfg = build_text_embedder_cfg(args, db, materialized_cache_dir, device)
                data, col_stats_dict = make_pkey_fkey_graph(
                    db,
                    col_to_stype_dict=col_to_stype_dict,
                    text_embedder_cfg=text_embedder_cfg,
                    cache_dir=str(materialized_cache_dir),
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            col_to_stype_dict = load_or_create_stypes(db, stypes_cache_path, rank)
            text_embedder_cfg = build_text_embedder_cfg(args, db, materialized_cache_dir, device)
            data, col_stats_dict = make_pkey_fkey_graph(
                db,
                col_to_stype_dict=col_to_stype_dict,
                text_embedder_cfg=text_embedder_cfg,
                cache_dir=str(materialized_cache_dir),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rank_print("Table names: ", list(db.table_dict.keys()))
        rank_print("Begin time: ", db.min_timestamp, "End time: ", db.max_timestamp)
        rank_print("Val time: ", dataset.val_timestamp, "Test time: ", dataset.test_timestamp)

        out_channels, loss_fn, tune_metric, higher_is_better, clamp_min, clamp_max = task_info(task)
        args.clamp_range = (clamp_min, clamp_max)

        loader_dict, entity_table = build_loader_dict(args, data, task, rank, world_size)
        rank_print("Entity table: ", entity_table)

        barrier()

        model = Model(
            data,
            col_stats_dict,
            args.num_layers,
            channels=args.channels,
            out_channels=out_channels,
            aggr=args.aggr,
            dropout=args.dropout,
            model_type=args.model_type,
            llm_frozen=args.llm_frozen,
            output_mlp=args.output_mlp,
            max_new_tokens=args.max_new_tokens,
            alpha=args.loss_class_weight,
            num_demo=args.num_demo,
            dataset=args.dataset,
            task=task,
            device=device,
        ).to(device)

        if is_distributed():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=(args.model_type != "gnn"),
            )

        model_for_io = unwrap_model(model)
        if args.wd != 0:
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [p for n, p in model.named_parameters() if "bias" not in n and "LayerNorm" not in n],
                        "weight_decay": args.wd,
                    },
                    {
                        "params": [p for n, p in model.named_parameters() if "bias" in n or "LayerNorm" in n],
                        "weight_decay": 0.0,
                    },
                ],
                lr=args.lr,
                betas=(0.9, 0.95),
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if higher_is_better else "min",
            factor=0.8,
            patience=100,
        )
        trainable_params, all_param = model_for_io.print_trainable_params()
        rank_print(
            f"trainable params: {trainable_params / 1e6:.2f}M || "
            f"all params: {all_param / 1e6:.2f}M || "
            f"trainable: {100 * trainable_params / all_param:.4f}%"
        )

        run = None
        state_dict = None

        if args.pretrain:
            if not args.debug and is_main_process():
                run = wandb.init(
                    project="rel-LLM-zero",
                    name=f"{args.dataset}_{args.task}",
                    id=f"pretrain_run_{args.dataset}_{args.task}",
                    resume="allow",
                )
            pretrain_steps = 0
            best_val_metric = -math.inf if higher_is_better else math.inf
            for epoch in range(1, args.pretrain_epochs + 1):
                loss_accum = count_accum = 0
                tq = tqdm(loader_dict["train"], total=len(loader_dict["train"]), disable=not is_main_process())
                for batch in tq:
                    try:
                        model.train()
                        batch = batch.to(device)
                        nums_samples = batch[entity_table].y.size(0)
                        optimizer.zero_grad()
                        loss = model(batch, task.entity_table, pretrain_mode=True)
                        loss.backward()
                        optimizer.step()
                    except torch.OutOfMemoryError:
                        rank_print("Skipping batch due to CUDA out of memory error")
                        torch.cuda.empty_cache()
                        continue

                    pretrain_steps += 1
                    loss_accum += loss.detach().item() * nums_samples
                    count_accum += nums_samples
                    train_loss = loss_accum / max(count_accum, 1)
                    summary = {"loss": train_loss, "lr": optimizer.param_groups[-1]["lr"]}
                    if run is not None:
                        for k, v in summary.items():
                            run.log({f"Pretrain/{k}": v}, step=pretrain_steps)
                    tq.set_description(
                        f"[Pretrain] Epoch/Step: {epoch:02d}/{pretrain_steps} | Train loss: {train_loss}"
                    )

                    if pretrain_steps % args.val_steps == 0:
                        demo = None
                        if args.num_demo > 0:
                            demo_batch = next(iter(loader_dict["train"])).to(device)
                            demo = model_for_io.get_demo_info(demo_batch, task.entity_table)
                        val_metrics = evaluate_loader(
                            loader_dict["val"],
                            model,
                            task,
                            args,
                            device,
                            demo,
                            max_steps=resolve_max_steps(args.eval_steps),
                            progress_desc="[Val]",
                        )
                        if run is not None:
                            for k, v in val_metrics.items():
                                run.log({f"val/{k}": v}, step=pretrain_steps)

                        improved = (
                            higher_is_better and val_metrics[tune_metric] >= best_val_metric
                        ) or (
                            not higher_is_better and val_metrics[tune_metric] <= best_val_metric
                        )
                        if improved:
                            best_val_metric = val_metrics[tune_metric]
                            state_dict = copy.deepcopy(model_for_io.state_dict())
                        scheduler.step(val_metrics[tune_metric])
                        rank_print(
                            f"[Eval] Epoch/Step: {epoch:02d}/{pretrain_steps} | "
                            f"Val: {val_metrics} | "
                            f"Best val: {best_val_metric:.4f}"
                        )
                        barrier()

        steps = 0
        if run is not None:
            run.finish()
            run = None
        if not args.debug and is_main_process():
            run = wandb.init(
                project="rel-LLM",
                name=f"{args.dataset}_{args.task}",
                id=f"finetune_run_{args.dataset}_{args.task}",
                resume="allow",
            )

        if state_dict is not None:
            model_for_io.load_state_dict(state_dict)

        best_val_metric = -math.inf if higher_is_better else math.inf
        while steps < args.train_steps:
            loss_accum = count_accum = 0
            remaining_steps = args.train_steps - steps
            tq = tqdm(
                loader_dict["train"],
                total=min(len(loader_dict["train"]), remaining_steps),
                disable=not is_main_process(),
            )
            for batch_idx, batch in enumerate(tq, start=1):
                model.train()
                batch = batch.to(device)
                nums_samples = batch[entity_table].y.size(0)
                optimizer.zero_grad()
                if args.model_type == "gnn" or args.output_mlp:
                    output_pred = model(batch, task.entity_table)
                    output_pred = (
                        output_pred.view(-1)
                        if len(output_pred.size()) > 1 and output_pred.size(1) == 1
                        else output_pred
                    )
                    loss = loss_fn(output_pred.float(), batch[entity_table].y.float())
                else:
                    loss = model(batch, task.entity_table)
                loss.backward()
                optimizer.step()

                steps += 1
                loss_accum += loss.detach().item() * nums_samples
                count_accum += nums_samples
                train_loss = loss_accum / max(count_accum, 1)
                summary = {"loss": train_loss, "lr": optimizer.param_groups[-1]["lr"]}
                if run is not None:
                    for k, v in summary.items():
                        run.log({f"train/{k}": v}, step=steps)
                current_train_step = min(batch_idx, remaining_steps)
                tq.set_description(
                    f"[Train] Step: {steps}/{args.train_steps} | "
                    f"Cycle step: {current_train_step}/{min(len(loader_dict['train']), remaining_steps)} | "
                    f"Train loss: {train_loss:3f}"
                )

                if steps % args.val_steps == 0:
                    val_metrics = evaluate_loader(
                        loader_dict["val"],
                        model,
                        task,
                        args,
                        device,
                        max_steps=resolve_max_steps(args.eval_steps),
                        progress_desc="[Val]",
                    )
                    if run is not None:
                        for k, v in val_metrics.items():
                            run.log({f"val/{k}": v}, step=steps)

                    improved = (
                        higher_is_better and val_metrics[tune_metric] >= best_val_metric
                    ) or (
                        not higher_is_better and val_metrics[tune_metric] <= best_val_metric
                    )
                    if improved:
                        best_val_metric = val_metrics[tune_metric]
                        state_dict = copy.deepcopy(model_for_io.state_dict())
                    scheduler.step(val_metrics[tune_metric])
                    rank_print(
                        f"[Eval] Train step: {steps}/{args.train_steps} | "
                        f"Eval cap: {args.eval_steps} step(s) | "
                        f"Val: {val_metrics} | "
                        f"Best val: {best_val_metric:.4f}"
                    )
                    barrier()
                if steps >= args.train_steps:
                    break

        if state_dict is None:
            state_dict = copy.deepcopy(model_for_io.state_dict())
        model_for_io.load_state_dict(state_dict)

        val_metrics = evaluate_loader(
            loader_dict["val"],
            model,
            task,
            args,
            device,
            max_steps=resolve_max_steps(args.eval_steps),
            progress_desc="[Val]",
        )
        rank_print(f"Best Val metrics: {val_metrics}")
        if not args.skip_test:
            test_metrics = evaluate_loader(
                loader_dict["test"],
                model,
                task,
                args,
                device,
                max_steps=resolve_max_steps(args.test_steps),
                progress_desc="[Test]",
            )
            rank_print(f"Best test metrics: {test_metrics}")
        if run is not None:
            for k, v in test_metrics.items():
                run.log({f"test/{k}": v}, step=steps + 1)
            run.finish()

        barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
