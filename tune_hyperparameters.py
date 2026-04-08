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
BEST_VAL_RE = re.compile(r"Best Val metrics: (\{.*\})")
BEST_TEST_RE = re.compile(r"Best test metrics: (\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-based hyperparameter tuning wrapper for main.py",
    )
    parser.add_argument("--dataset", type=str, default="rel-amazon")
    parser.add_argument("--task", type=str, default="user-churn")
    parser.add_argument("--study-name", type=str, default=None)
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
        "--nccl-p2p-disable",
        type=str,
        default="1",
        help="Export NCCL_P2P_DISABLE for each trial. Default is 1.",
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--text-embedder-path", type=str, default=None)
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
        f"--test_steps={args.test_steps}",
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
    }
    return build_main_command(args, params, skip_test=True), params


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

                    metrics = safe_parse_metrics(line, EVAL_LINE_RE)
                    if metrics is not None and tune_metric in metrics:
                        eval_step += 1
                        current_metric = float(metrics[tune_metric])
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

        for line in reversed(all_output):
            metrics = safe_parse_metrics(line, BEST_VAL_RE)
            if metrics is not None and tune_metric in metrics:
                best_metric = float(metrics[tune_metric])
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
    study: optuna.Study,
    output_dir: Path,
) -> dict[str, Any]:
    best_params = study.best_trial.params
    command = build_main_command(args, best_params, skip_test=False)
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
    best_test_metrics = None
    for line in reversed(all_output):
        if best_test_metrics is None:
            best_test_metrics = safe_parse_metrics(line, BEST_TEST_RE)
        if best_val_metrics is None:
            best_val_metrics = safe_parse_metrics(line, BEST_VAL_RE)
        if best_val_metrics is not None and best_test_metrics is not None:
            break

    return {
        "command": " ".join(command),
        "log_path": str(log_path),
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
    }


def save_best_result(
    study: optuna.Study,
    output_dir: Path,
    final_test_result: dict[str, Any] | None = None,
) -> None:
    best = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
    }
    if final_test_result is not None:
        best["final_test"] = final_test_result
    with (output_dir / "best_trial.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    study_name = args.study_name or f"{args.dataset}_{args.task}_tuning"
    output_dir = Path(args.output_dir) / study_name
    reset_study_if_requested(args, study_name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tune_metric, higher_is_better = get_tuning_target(args.dataset, args.task)
    direction = "maximize" if higher_is_better else "minimize"

    sampler = TPESampler(seed=args.seed, n_startup_trials=args.startup_trials)
    pruner = MedianPruner(
        n_startup_trials=args.startup_trials,
        n_warmup_steps=args.prune_warmup_steps,
    )
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
    final_test_result = run_final_test(args, study, output_dir)
    save_best_result(study, output_dir, final_test_result)

    print(f"Study: {study.study_name}")
    print(f"Direction: {direction}")
    print(f"Best {tune_metric}: {study.best_value}")
    print(f"Best params: {study.best_params}")
    print(f"Saved best trial to: {output_dir / 'best_trial.json'}")


if __name__ == "__main__":
    main()
