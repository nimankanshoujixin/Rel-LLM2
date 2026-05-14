import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from relbench.tasks import get_task
from utils import task_info

EVAL_LINE_RE = re.compile(r"\[Eval\].*?\| Val: (\{.*?\})(?: \| |$)")
STAGE3_EVAL_LINE_RE = re.compile(
    r"\[Eval\].*?\| Val: (?P<val>\{.*?\}) \| "
    r"TestSubset: (?P<test>\{.*?\}|None) \| "
    r"SelectionSource: (?P<source>\w+) \| "
    r"Best selection metric: (?P<best>[-+]?\d*\.?\d+)"
)
BEST_VAL_RE = re.compile(r"Best Val metrics: (\{.*\})")
BEST_TEST_SUBSET_RE = re.compile(r"Best TestSubset metrics: (\{.*\})")
BEST_TEST_RE = re.compile(r"Best test metrics: (\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter tuning wrapper for main.py",
    )
    parser.add_argument("--dataset", type=str, default="rel-amazon")
    parser.add_argument("--task", type=str, default="user-churn")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument(
        "--final-test-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Optuna search and run final confirmation only from an existing study.",
    )
    parser.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete the existing study with the same name and remove its output directory before running.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///optuna_rel_llm.db",
        help="Optuna storage URL.",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--startup-trials", type=int, default=8)
    parser.add_argument("--prune-warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2**15)
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=1024)
    parser.add_argument("--test-steps", type=int, default=4096)
    parser.add_argument(
        "--final-test-steps",
        type=int,
        default=-1,
        help="Final confirmation test cap for the best trial. Use -1 for full test.",
    )
    parser.add_argument(
        "--run-final-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run final confirmation on the selected best trial after Optuna completes.",
    )
    parser.add_argument(
        "--periodic-test-steps",
        type=int,
        default=0,
        help="Subset test batches evaluated during training. Set >0 to enable Stage 3 screening protocol.",
    )
    parser.add_argument(
        "--model-selection-source",
        type=str,
        default="val",
        choices=["val", "test_subset"],
        help="Checkpoint selection source passed through to main.py.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-metric-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-loss-delta", type=float, default=0.0)
    parser.add_argument("--val-size", type=int, default=1)
    parser.add_argument(
        "--batch-size-choices",
        type=str,
        default="1,2,4",
        help="Comma-separated batch size candidates.",
    )
    parser.add_argument(
        "--channels-choices",
        type=str,
        default="64,128,256",
        help="Comma-separated channel candidates.",
    )
    parser.add_argument(
        "--num-layers-choices",
        type=str,
        default="1,2,3",
        help="Comma-separated layer count candidates.",
    )
    parser.add_argument(
        "--num-neighbors-choices",
        type=str,
        default="16,32,64,128",
        help="Comma-separated neighbor count candidates.",
    )
    parser.add_argument(
        "--aggr-choices",
        type=str,
        default="sum,mean",
        help="Comma-separated aggregation candidates.",
    )
    parser.add_argument(
        "--temporal-strategy-choices",
        type=str,
        default="uniform,last",
        help="Comma-separated temporal strategy candidates.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch main.py or torch.distributed.run.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Use DDP launch when greater than 1.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Master port used by torchrun in DDP mode.",
    )
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=None,
        help="If set, exports CUDA_VISIBLE_DEVICES to this value for each trial.",
    )
    parser.add_argument(
        "--max-gpus-per-task",
        type=int,
        default=None,
        help="Optional cap for GPUs used by one Optuna task. Useful for keeping tuning batch granularity fine.",
    )
    parser.add_argument(
        "--nccl-p2p-disable",
        type=str,
        default="1",
        help="Export NCCL_P2P_DISABLE for each trial. Default is 1.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--text-embedder-path", type=str, default=None)
    parser.add_argument(
        "--basis-root",
        type=str,
        default="artifacts/basis",
        help="Root directory for per-dataset basis artifacts.",
    )
    parser.add_argument(
        "--basis-artifact",
        type=str,
        default=None,
        help="Explicit basis artifact path. Overrides --basis-root when set.",
    )
    parser.add_argument(
        "--gnn-repr-artifact",
        type=str,
        default=None,
        help="Optional GNN representation artifact path for artifact-backed finetune.",
    )
    parser.add_argument(
        "--disable-basis-token-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable the impl-b basis token alignment head.",
    )
    parser.add_argument(
        "--basis-tau",
        type=float,
        default=None,
        help="Fixed temperature used for token-to-basis logits. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-residual-alpha",
        type=float,
        default=None,
        help="Fixed residual injection scale. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-graph-alpha",
        type=float,
        default=None,
        help="Fixed global graph residual scale. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-lambda-tok",
        type=float,
        default=None,
        help="Fixed token-level BCE alignment weight. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-lambda-g",
        type=float,
        default=None,
        help="Fixed graph-level BCE alignment weight. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-lambda-sharp",
        type=float,
        default=None,
        help="Fixed entropy sharpening weight. If omitted, tune it with Optuna.",
    )
    parser.add_argument(
        "--basis-lambda-postalign-tok",
        type=float,
        default=0.0,
        help="Fixed post-alignment token retention weight.",
    )
    parser.add_argument(
        "--basis-lambda-entity-identity",
        type=float,
        default=0.0,
        help="Fixed entity-identity contrastive weight.",
    )
    parser.add_argument(
        "--basis-entity-identity-temperature",
        type=float,
        default=0.1,
        help="Fixed entity-identity temperature.",
    )
    parser.add_argument(
        "--basis-lambda-branch-orth",
        type=float,
        default=0.0,
        help="Fixed branch orthogonality regularization weight.",
    )
    parser.add_argument(
        "--basis-gate-strategy",
        type=str,
        default="none",
        choices=["none", "confidence"],
        help="Fixed conservative transfer gate strategy.",
    )
    parser.add_argument(
        "--basis-gate-token-floor",
        type=float,
        default=0.0,
        help="Fixed token-confidence gate floor.",
    )
    parser.add_argument(
        "--basis-gate-graph-floor",
        type=float,
        default=0.0,
        help="Fixed graph-confidence gate floor.",
    )
    parser.add_argument(
        "--basis-assignment-topk",
        type=int,
        default=0,
        help="Fixed sparse top-k basis assignment. Use 0 to disable.",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Working directory for trial subprocesses.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_runs",
        help="Directory to store trial logs and best params.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--text-embedder",
        type=str,
        default="mpnet",
        choices=["glove", "mpnet"],
    )
    parser.add_argument(
        "--llm-frozen",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-mlp",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pretrain",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --debug to main.py to disable wandb during tuning.",
    )
    return parser.parse_args()


def parse_int_choices(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_str_choices(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_gpu_id_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def apply_optuna_gpu_cap(args: argparse.Namespace) -> None:
    if args.max_gpus_per_task is None:
        return
    if args.max_gpus_per_task < 1:
        raise ValueError("--max-gpus-per-task must be at least 1 when provided.")

    capped_gpu_ids = normalize_gpu_id_list(args.gpu_id)
    if capped_gpu_ids:
        capped_gpu_ids = capped_gpu_ids[: args.max_gpus_per_task]
        args.gpu_id = ",".join(capped_gpu_ids)
        args.nproc_per_node = min(args.nproc_per_node, len(capped_gpu_ids))
    else:
        args.nproc_per_node = min(args.nproc_per_node, args.max_gpus_per_task)

    if args.nproc_per_node < 1:
        args.nproc_per_node = 1


def safe_parse_metrics(line: str, regex: re.Pattern[str]) -> dict[str, float] | None:
    match = regex.search(line)
    if match is None:
        return None
    try:
        metrics = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(metrics, dict):
        return None
    return metrics


def parse_stage3_eval_metrics(
    line: str,
) -> tuple[dict[str, float], dict[str, float] | None, str] | None:
    match = STAGE3_EVAL_LINE_RE.search(line)
    if match is None:
        return None
    try:
        val_metrics = ast.literal_eval(match.group("val"))
        test_blob = match.group("test")
        test_metrics = None if test_blob == "None" else ast.literal_eval(test_blob)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(val_metrics, dict):
        return None
    if test_metrics is not None and not isinstance(test_metrics, dict):
        return None
    return val_metrics, test_metrics, match.group("source")


def reset_study_if_requested(
    args: argparse.Namespace,
    study_name: str,
    output_dir: Path,
) -> None:
    if not args.reset_study:
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        optuna.delete_study(study_name=study_name, storage=args.storage)
    except KeyError:
        pass


def build_main_command(
    args: argparse.Namespace,
    params: dict[str, Any],
    *,
    skip_test: bool,
    test_steps_override: int | None = None,
) -> list[str]:
    main_command = [
        "main.py",
        f"--dataset={args.dataset}",
        f"--task={args.task}",
        f"--model_type={args.model_type}",
        f"--text_embedder={args.text_embedder}",
        f"--train_steps={args.train_steps}",
        f"--val_steps={args.val_steps}",
        f"--eval_steps={args.eval_steps}",
        f"--test_steps={args.test_steps if test_steps_override is None else test_steps_override}",
        f"--periodic_test_steps={args.periodic_test_steps}",
        f"--model_selection_source={args.model_selection_source}",
        f"--early_stop_patience={args.early_stop_patience}",
        f"--early_stop_metric_delta={args.early_stop_metric_delta}",
        f"--early_stop_loss_delta={args.early_stop_loss_delta}",
        f"--val_size={args.val_size}",
        f"--channels={params['channels']}",
        f"--num_layers={params['num_layers']}",
        f"--num_neighbors={params['num_neighbors']}",
        f"--aggr={params['aggr']}",
        f"--temporal_strategy={params['temporal_strategy']}",
        f"--dropout={params['dropout']}",
        f"--batch_size={params['batch_size']}",
        f"--lr={params['lr']}",
        f"--wd={params['wd']}",
        f"--seed={args.seed}",
        f"--basis_root={args.basis_root}",
        f"--basis_tau={params['basis_tau']}",
        f"--basis_residual_alpha={params['basis_residual_alpha']}",
        f"--basis_graph_alpha={params['basis_graph_alpha']}",
        f"--basis_lambda_tok={params['basis_lambda_tok']}",
        f"--basis_lambda_g={params['basis_lambda_g']}",
        f"--basis_lambda_sharp={params['basis_lambda_sharp']}",
        f"--basis_lambda_postalign_tok={args.basis_lambda_postalign_tok}",
        f"--basis_lambda_entity_identity={args.basis_lambda_entity_identity}",
        f"--basis_entity_identity_temperature={args.basis_entity_identity_temperature}",
        f"--basis_lambda_branch_orth={args.basis_lambda_branch_orth}",
        f"--basis_gate_strategy={args.basis_gate_strategy}",
        f"--basis_gate_token_floor={args.basis_gate_token_floor}",
        f"--basis_gate_graph_floor={args.basis_gate_graph_floor}",
        f"--basis_assignment_topk={args.basis_assignment_topk}",
        "--loss_class_weight",
        "1.0",
        str(params["w_pos"]),
    ]

    if args.nproc_per_node > 1:
        command = [
            args.python_executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            *main_command,
        ]
    else:
        command = [args.python_executable, *main_command]

    if args.cache_dir:
        command.append(f"--cache_dir={args.cache_dir}")
    if args.text_embedder_path:
        command.append(f"--text_embedder_path={args.text_embedder_path}")
    if args.basis_artifact:
        command.append(f"--basis_artifact={args.basis_artifact}")
    if args.gnn_repr_artifact:
        command.append(f"--gnn_repr_artifact={args.gnn_repr_artifact}")
    if args.disable_basis_token_head:
        command.append("--disable_basis_token_head")
    if args.llm_frozen:
        command.append("--llm_frozen")
    if args.output_mlp:
        command.append("--output_mlp")
    if args.pretrain:
        command.append("--pretrain")
    if args.debug:
        command.append("--debug")
    if skip_test:
        command.append("--skip_test")

    return command


def build_trial_command(args: argparse.Namespace, trial: optuna.Trial) -> tuple[list[str], dict[str, Any]]:
    batch_size_choices = parse_int_choices(args.batch_size_choices)
    channels_choices = parse_int_choices(args.channels_choices)
    num_layers_choices = parse_int_choices(args.num_layers_choices)
    num_neighbors_choices = parse_int_choices(args.num_neighbors_choices)
    aggr_choices = parse_str_choices(args.aggr_choices)
    temporal_choices = parse_str_choices(args.temporal_strategy_choices)

    params = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "wd": trial.suggest_float("wd", 1e-6, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "channels": trial.suggest_categorical("channels", channels_choices),
        "num_layers": trial.suggest_categorical("num_layers", num_layers_choices),
        "num_neighbors": trial.suggest_categorical("num_neighbors", num_neighbors_choices),
        "aggr": trial.suggest_categorical("aggr", aggr_choices),
        "temporal_strategy": trial.suggest_categorical(
            "temporal_strategy",
            temporal_choices,
        ),
        "batch_size": trial.suggest_categorical("batch_size", batch_size_choices),
        "w_pos": trial.suggest_float("w_pos", 0.5, 3.0),
        "basis_residual_alpha": (
            args.basis_residual_alpha
            if args.basis_residual_alpha is not None
            else trial.suggest_float("basis_residual_alpha", 0.05, 0.5)
        ),
        "basis_graph_alpha": (
            args.basis_graph_alpha
            if args.basis_graph_alpha is not None
            else trial.suggest_float("basis_graph_alpha", 0.0, 0.25)
        ),
        "basis_tau": (
            args.basis_tau
            if args.basis_tau is not None
            else trial.suggest_float("basis_tau", 0.03, 0.2, log=True)
        ),
        "basis_lambda_tok": (
            args.basis_lambda_tok
            if args.basis_lambda_tok is not None
            else trial.suggest_float("basis_lambda_tok", 0.1, 3.0, log=True)
        ),
        "basis_lambda_g": (
            args.basis_lambda_g
            if args.basis_lambda_g is not None
            else trial.suggest_float("basis_lambda_g", 0.1, 3.0, log=True)
        ),
        "basis_lambda_sharp": (
            args.basis_lambda_sharp
            if args.basis_lambda_sharp is not None
            else trial.suggest_float("basis_lambda_sharp", 1e-4, 1e-1, log=True)
        ),
    }
    skip_test = args.model_selection_source != "test_subset" and args.periodic_test_steps <= 0
    return build_main_command(args, params, skip_test=skip_test), params


def get_tuning_target(dataset_name: str, task_name: str) -> tuple[str, bool]:
    task = get_task(dataset_name, task_name, download=False)
    _, _, tune_metric, higher_is_better, _, _ = task_info(task)
    return tune_metric, higher_is_better


def terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def objective_factory(
    args: argparse.Namespace,
    tune_metric: str,
    higher_is_better: bool,
    trial_dir: Path,
):
    def objective(trial: optuna.Trial) -> float:
        command, sampled_params = build_trial_command(args, trial)
        trial.set_user_attr("command", " ".join(command))
        for key, value in sampled_params.items():
            trial.set_user_attr(key, value)

        log_path = trial_dir / f"trial_{trial.number:04d}.log"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if args.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        env["NCCL_P2P_DISABLE"] = args.nccl_p2p_disable

        best_metric: float | None = None
        eval_step = 0
        all_output: list[str] = []

        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=args.workdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.write(line)
                    log_file.write(line)
                    log_file.flush()
                    all_output.append(line)

                    selected_metrics = None
                    stage3_eval = parse_stage3_eval_metrics(line)
                    if stage3_eval is not None:
                        val_metrics, test_metrics, selection_source = stage3_eval
                        if (
                            selection_source == "test_subset"
                            and test_metrics is not None
                            and tune_metric in test_metrics
                        ):
                            selected_metrics = test_metrics
                        elif selection_source == "val" and tune_metric in val_metrics:
                            selected_metrics = val_metrics
                    else:
                        metrics = safe_parse_metrics(line, EVAL_LINE_RE)
                        if metrics is not None and tune_metric in metrics:
                            selected_metrics = metrics

                    if selected_metrics is not None and tune_metric in selected_metrics:
                        eval_step += 1
                        current_metric = float(selected_metrics[tune_metric])
                        best_metric = current_metric
                        trial.report(current_metric, step=eval_step)
                        if eval_step >= args.prune_warmup_steps and trial.should_prune():
                            terminate_process(proc)
                            raise optuna.TrialPruned(
                                f"Pruned at eval step {eval_step} with "
                                f"{tune_metric}={current_metric:.6f}"
                            )

                return_code = proc.wait()
            finally:
                if proc.poll() is None:
                    terminate_process(proc)

        full_output = "".join(all_output)
        if return_code != 0:
            lowered = full_output.lower()
            if "out of memory" in lowered or "cuda out of memory" in lowered:
                raise optuna.TrialPruned("Trial hit CUDA OOM.")
            raise RuntimeError(f"Trial failed with exit code {return_code}.")

        best_metric_regexes = [BEST_VAL_RE]
        if args.model_selection_source == "test_subset":
            best_metric_regexes.insert(0, BEST_TEST_SUBSET_RE)

        for regex in best_metric_regexes:
            for line in reversed(all_output):
                metrics = safe_parse_metrics(line, regex)
                if metrics is not None and tune_metric in metrics:
                    best_metric = float(metrics[tune_metric])
                    break
            if best_metric is not None:
                break

        if best_metric is None:
            raise RuntimeError(
                f"Unable to parse '{tune_metric}' from trial output. "
                f"Check log: {log_path}"
            )

        return best_metric

    return objective


def run_final_test(
    args: argparse.Namespace,
    best_params: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    command = build_main_command(
        args,
        best_params,
        skip_test=False,
        test_steps_override=args.final_test_steps,
    )
    log_path = output_dir / "best_trial_test.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    env["NCCL_P2P_DISABLE"] = args.nccl_p2p_disable

    all_output: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=args.workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_file.write(line)
                log_file.flush()
                all_output.append(line)
            return_code = proc.wait()
        finally:
            if proc.poll() is None:
                terminate_process(proc)

    if return_code != 0:
        raise RuntimeError(f"Best-trial final test failed with exit code {return_code}.")

    best_val_metrics = None
    best_test_subset_metrics = None
    best_test_metrics = None
    for line in reversed(all_output):
        if best_test_metrics is None:
            best_test_metrics = safe_parse_metrics(line, BEST_TEST_RE)
        if best_test_subset_metrics is None:
            best_test_subset_metrics = safe_parse_metrics(line, BEST_TEST_SUBSET_RE)
        if best_val_metrics is None:
            best_val_metrics = safe_parse_metrics(line, BEST_VAL_RE)
        if (
            best_val_metrics is not None
            and best_test_metrics is not None
            and (args.periodic_test_steps <= 0 or best_test_subset_metrics is not None)
        ):
            break

    return {
        "command": " ".join(command),
        "log_path": str(log_path),
        "best_val_metrics": best_val_metrics,
        "best_test_subset_metrics": best_test_subset_metrics,
        "best_test_metrics": best_test_metrics,
    }


def save_best_result(
    record: dict[str, Any],
    output_dir: Path,
    final_test_result: dict[str, Any] | None = None,
) -> None:
    best = dict(record)
    if final_test_result is not None:
        best["final_test"] = final_test_result
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    apply_optuna_gpu_cap(args)
    study_name = args.study_name or f"{args.dataset}_{args.task}_tuning"
    output_dir = Path(args.output_dir) / study_name

    tune_metric, higher_is_better = get_tuning_target(args.dataset, args.task)
    direction = "maximize" if higher_is_better else "minimize"

    sampler = TPESampler(seed=args.seed, n_startup_trials=args.startup_trials)
    pruner = MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.prune_warmup_steps,
    )

    if args.final_test_only:
        if args.reset_study:
            raise ValueError("--final-test-only cannot be combined with --reset-study.")
        output_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any]
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=args.storage,
                sampler=sampler,
                pruner=pruner,
            )
            record = {
                "study_name": study.study_name,
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
            }
        except KeyError:
            best_trial_path = output_dir / "best_trial.json"
            if not best_trial_path.exists():
                raise
            with best_trial_path.open("r", encoding="utf-8") as f:
                record = json.load(f)
            if not isinstance(record, dict) or "best_params" not in record:
                raise ValueError(f"Invalid best trial JSON: {best_trial_path}")
        final_test_result = run_final_test(args, record["best_params"], output_dir)
        save_best_result(record, output_dir, final_test_result)
        print(f"Study: {record.get('study_name', study_name)}")
        print(f"Direction: {direction}")
        print(f"Best {tune_metric}: {record.get('best_value')}")
        print(f"Best params: {record.get('best_params')}")
        print(f"Saved final test log to: {output_dir / 'best_trial_test.log'}")
        print(f"Saved best trial to: {output_dir / 'best_trial.json'}")
        return

    reset_study_if_requested(args, study_name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    objective = objective_factory(
        args=args,
        tune_metric=tune_metric,
        higher_is_better=higher_is_better,
        trial_dir=output_dir,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    final_test_result = None
    if args.run_final_test:
        final_test_result = run_final_test(args, study.best_trial.params, output_dir)
    save_best_result(
        {
            "study_name": study.study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_user_attrs": study.best_trial.user_attrs,
        },
        output_dir,
        final_test_result,
    )

    print(f"Study: {study.study_name}")
    print(f"Direction: {direction}")
    print(f"Best {tune_metric}: {study.best_value}")
    print(f"Best params: {study.best_params}")
    if args.run_final_test:
        print(f"Saved final test log to: {output_dir / 'best_trial_test.log'}")
    print(f"Saved best trial to: {output_dir / 'best_trial.json'}")


if __name__ == "__main__":
    main()
