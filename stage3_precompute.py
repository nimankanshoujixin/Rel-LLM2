from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from stage3_research import (
    PIPELINE_CONFIG_PATH,
    ROOT,
    SshTarget,
    build_ssh_targets,
    dump_json,
    get_target_by_name,
    load_json,
    normalize_gpu_spec,
    probe_remote_targets,
    query_tmux_windows,
    run_subprocess,
    sanitize_slug,
    ssh_command,
)


PRECOMPUTE_DIR = ROOT / "stage3_notes" / "precompute_jobs"


def utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def precompute_job_path(job_id: str) -> Path:
    return PRECOMPUTE_DIR / f"{job_id}.json"


def write_shell_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def compact_log_tail(text: str, max_lines: int = 20) -> str:
    normalized = text.replace("\r", "\n")
    raw_lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    compacted: list[str] = []
    pending_progress: str | None = None
    progress_prefixes = (
        "Embedding raw data in mini-batch:",
        "Batches:",
    )

    for line in raw_lines:
        if line.startswith(progress_prefixes):
            pending_progress = line
            continue
        if pending_progress is not None:
            compacted.append(pending_progress)
            pending_progress = None
        compacted.append(line)

    if pending_progress is not None:
        compacted.append(pending_progress)

    if len(compacted) > max_lines:
        compacted = compacted[-max_lines:]
    return "\n".join(compacted)


def scp_recursive_to_remote(target: SshTarget, local_paths: list[Path], remote_dir: str) -> subprocess.CompletedProcess[str]:
    command = ["scp", "-r", "-P", str(target.port)]
    command.extend(str(path) for path in local_paths)
    command.append(f"{target.destination}:{remote_dir}/")
    return run_subprocess(command, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch and monitor long Stage 3 precompute jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_cache = subparsers.add_parser("launch-cache", help="Launch a graph-cache precompute job.")
    launch_cache.add_argument("--dataset", required=True)
    launch_cache.add_argument("--target", default=None)
    launch_cache.add_argument("--gpus", default=None, help="Comma-separated GPU indices. Default: all idle GPUs on target.")
    launch_cache.add_argument(
        "--cache-profile",
        default="gnn_repr_train_visible",
        choices=["task_default", "autocomplete_masked", "gnn_repr_train_visible", "gnn_repr_full_horizon"],
    )
    launch_cache.add_argument("--text-embedder", default="mpnet", choices=["glove", "mpnet"])
    launch_cache.add_argument("--text-embedder-path", default="./cache")
    launch_cache.add_argument("--cache-dir", default="~/.cache/relbench_examples")
    launch_cache.add_argument("--embed-batch-size", type=int, default=256)
    launch_cache.add_argument("--embed-chunk-size", type=int, default=0)
    launch_cache.add_argument(
        "--force-cpu-aggregate",
        action="store_true",
        help="Force CPU-side aggregation/materialization after text embedding.",
    )
    launch_cache.add_argument(
        "--table-names",
        default="",
        help="Optional comma-separated subset of tables to materialize.",
    )
    launch_cache.add_argument(
        "--parallelism-mode",
        default="table_shards",
        choices=["table_shards", "embed_multi_gpu"],
        help="Use per-table shards across GPUs or a single process with multi-GPU text embedding.",
    )
    launch_cache.add_argument("--download", action="store_true")
    launch_cache.add_argument("--copy-from-equivalent", action="store_true")
    launch_cache.add_argument("--skip-existing", action="store_true")
    launch_cache.add_argument("--job-suffix", default="")

    launch_gnn = subparsers.add_parser("launch-gnn-repr", help="Launch a GNN representation artifact build job.")
    launch_gnn.add_argument("--dataset", required=True)
    launch_gnn.add_argument("--target", default=None)
    launch_gnn.add_argument("--gpu", required=True, type=int)
    launch_gnn.add_argument("--text-embedder", default="mpnet", choices=["glove", "mpnet"])
    launch_gnn.add_argument("--train-steps", type=int, default=1024)
    launch_gnn.add_argument("--batch-size", type=int, default=256)
    launch_gnn.add_argument("--num-neighbors", type=int, default=64)
    launch_gnn.add_argument("--channels", type=int, default=256)
    launch_gnn.add_argument("--num-layers", type=int, default=2)
    launch_gnn.add_argument("--aggr", type=str, default="mean")
    launch_gnn.add_argument("--download", action="store_true")
    launch_gnn.add_argument("--job-suffix", default="")

    launch_basis = subparsers.add_parser("launch-basis", help="Launch a basis artifact build job.")
    launch_basis.add_argument("--dataset", required=True)
    launch_basis.add_argument("--target", default=None)
    launch_basis.add_argument("--gpu", required=True, type=int)
    launch_basis.add_argument("--model-type", default="./Llama-3.2-1B")
    launch_basis.add_argument("--db-scope", default="train_visible", choices=["train_visible", "upto_test", "full"])
    launch_basis.add_argument("--download", action="store_true")
    launch_basis.add_argument("--job-suffix", default="")

    status = subparsers.add_parser("status", help="Show tmux/log status for a precompute job.")
    status.add_argument("job", type=Path)
    status.add_argument("--tail-lines", type=int, default=20)

    list_jobs = subparsers.add_parser("list-jobs", help="List known precompute jobs.")
    list_jobs.set_defaults(command="list-jobs")
    return parser.parse_args()


def default_target_name(config: dict[str, Any]) -> str:
    return build_ssh_targets(config)[0].name


def pick_idle_gpus(config: dict[str, Any], target_name: str) -> list[int]:
    targets = build_ssh_targets(config)
    snapshot = probe_remote_targets(targets, config)
    rows = snapshot[target_name]["idle_gpus"]
    return [int(row["index"]) for row in rows]


def sync_common_runtime(target: SshTarget, kind: str) -> None:
    base_paths = [ROOT / "main.py", ROOT / "model.py", ROOT / "utils.py", ROOT / "text_embedder.py"]
    if kind in {"graph_cache", "gnn_repr"}:
        base_paths.append(ROOT / "precompute")
    if kind == "gnn_repr":
        base_paths.append(ROOT / "gnn_repr")
    if kind == "basis":
        base_paths.append(ROOT / "basis")

    mkdir_proc = ssh_command(target, f"mkdir -p {target.remote_repo_root} {target.remote_tmp_dir}")
    if mkdir_proc.returncode != 0:
        raise RuntimeError(f"Failed to prepare remote directories on {target.name}: {mkdir_proc.stderr.strip()}")
    sync_proc = scp_recursive_to_remote(target, base_paths, target.remote_repo_root)
    if sync_proc.returncode != 0:
        raise RuntimeError(f"Code sync failed for {target.name}: {sync_proc.stderr.strip()}")


def launch_job(job: dict[str, Any]) -> None:
    config = load_json(PIPELINE_CONFIG_PATH)
    target = get_target_by_name(build_ssh_targets(config), job["target"])
    sync_common_runtime(target, job["kind"])
    script_paths: list[Path] = []
    for window in job["windows"]:
        local_script = ROOT / ".codex_remote" / "precompute" / f"{window['name']}.sh"
        write_shell_script(local_script, window["script_lines"])
        script_paths.append(local_script)
        window["local_script"] = str(local_script)
    upload_proc = scp_recursive_to_remote(target, script_paths, target.remote_tmp_dir)
    if upload_proc.returncode != 0:
        raise RuntimeError(f"Wrapper sync failed for {target.name}: {upload_proc.stderr.strip()}")

    for window in job["windows"]:
        remote_script = window["remote_script"]
        remote_log = window["remote_log"]
        proc = ssh_command(
            target,
            " && ".join(
                [
                    f"tmux has-session -t {target.tmux_session} >/dev/null 2>&1 || tmux new-session -d -s {target.tmux_session}",
                    f"chmod +x {remote_script}",
                    f"tmux kill-window -t {target.tmux_session}:{window['name']} >/dev/null 2>&1 || true",
                    f": > {remote_log}",
                    f"tmux new-window -d -t {target.tmux_session}: -n {window['name']} \"sh -lc 'bash {remote_script} > {remote_log} 2>&1'\"",
                ]
            ),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to launch {window['name']} on {target.name}: {proc.stderr.strip()}")

    job["status"] = "running"
    dump_json(precompute_job_path(job["job_id"]), job)


def build_graph_cache_job(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    target_name = args.target or default_target_name(config)
    gpu_list = normalize_gpu_spec(args.gpus.split(",")) if args.gpus else pick_idle_gpus(config, target_name)
    if not gpu_list:
        raise RuntimeError(f"No idle GPUs available for graph-cache job on target {target_name}.")
    if args.copy_from_equivalent:
        gpu_list = gpu_list[:1]
    shard_count = len(gpu_list)
    slug = sanitize_slug(f"{args.dataset}_{args.cache_profile}_{args.job_suffix or 'cache'}")
    job_id = f"precompute_{slug}_{utc_stamp().lower()}"
    target = get_target_by_name(build_ssh_targets(config), target_name)

    windows = []
    if args.parallelism_mode == "embed_multi_gpu":
        shard_output_root = f"{target.remote_tmp_dir}/{sanitize_slug(args.dataset)}_{sanitize_slug(args.cache_profile)}_{utc_stamp().lower()}_row_shards"
        for row_shard_index, gpu in enumerate(gpu_list):
            window_name = f"stage3-cache-{sanitize_slug(args.dataset)}-row{row_shard_index}"
            remote_script = f"{target.remote_tmp_dir}/{window_name}.sh"
            remote_log = f"{target.remote_tmp_dir}/{window_name}.log"
            command = [
                target.remote_python,
                "-u",
                "-m",
                "precompute.graph_cache",
                f"--dataset={args.dataset}",
                f"--cache-profile={args.cache_profile}",
                f"--cache_dir={args.cache_dir}",
                f"--text_embedder={args.text_embedder}",
                f"--text_embedder_path={args.text_embedder_path}",
                f"--embed_batch_size={args.embed_batch_size}",
                f"--device=cuda:0",
                "--num_shards=1",
                "--shard_index=0",
                f"--row_shard_count={len(gpu_list)}",
                f"--row_shard_index={row_shard_index}",
                f"--shard_output_root={shard_output_root}",
            ]
            if args.table_names:
                command.append(f"--table_names={args.table_names}")
            if args.embed_chunk_size > 0:
                command.append(f"--embed_chunk_size={args.embed_chunk_size}")
            if args.force_cpu_aggregate:
                command.append("--force_cpu_aggregate")
            if args.download:
                command.append("--download")
            if args.copy_from_equivalent:
                command.append("--copy_from_equivalent")
            if args.skip_existing:
                command.append("--skip_existing")
            script_lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {target.remote_repo_root}",
                f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command),
            ]
            windows.append(
                {
                    "name": window_name,
                    "gpu": gpu,
                    "row_shard_index": row_shard_index,
                    "row_shard_count": len(gpu_list),
                    "parallelism_mode": args.parallelism_mode,
                    "remote_script": remote_script,
                    "remote_log": remote_log,
                    "script_lines": script_lines,
                }
            )
        merge_window_name = f"stage3-cache-{sanitize_slug(args.dataset)}-merge"
        merge_remote_script = f"{target.remote_tmp_dir}/{merge_window_name}.sh"
        merge_remote_log = f"{target.remote_tmp_dir}/{merge_window_name}.log"
        worker_names = [window["name"] for window in windows]
        merge_command = [
            target.remote_python,
            "-u",
            "-m",
            "precompute.graph_cache",
            f"--dataset={args.dataset}",
            f"--cache-profile={args.cache_profile}",
            f"--cache_dir={args.cache_dir}",
            f"--text_embedder={args.text_embedder}",
            f"--text_embedder_path={args.text_embedder_path}",
            f"--embed_batch_size={args.embed_batch_size}",
            "--device=cpu",
            "--num_shards=1",
            "--shard_index=0",
            "--merge_shards_only",
            f"--row_shard_count={len(gpu_list)}",
            f"--shard_output_root={shard_output_root}",
        ]
        if args.table_names:
            merge_command.append(f"--table_names={args.table_names}")
        if args.skip_existing:
            merge_command.append("--skip_existing")
        merge_script_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {target.remote_repo_root}",
            "while true; do",
            f"  active=0; for name in {' '.join(worker_names)}; do tmux list-windows -t {target.tmux_session} 2>/dev/null | grep -q \"$name\" && active=1; done",
            "  if [ \"$active\" -eq 0 ]; then break; fi",
            "  sleep 20",
            "done",
            " ".join(merge_command),
        ]
        windows.append(
            {
                "name": merge_window_name,
                "gpu": None,
                "role": "merge",
                "parallelism_mode": args.parallelism_mode,
                "remote_script": merge_remote_script,
                "remote_log": merge_remote_log,
                "script_lines": merge_script_lines,
            }
        )
    else:
        for shard_index, gpu in enumerate(gpu_list):
            window_name = f"stage3-cache-{sanitize_slug(args.dataset)}-{shard_index}"
            remote_script = f"{target.remote_tmp_dir}/{window_name}.sh"
            remote_log = f"{target.remote_tmp_dir}/{window_name}.log"
            command = [
                target.remote_python,
                "-u",
                "-m",
                "precompute.graph_cache",
                f"--dataset={args.dataset}",
                f"--cache-profile={args.cache_profile}",
                f"--cache_dir={args.cache_dir}",
                f"--text_embedder={args.text_embedder}",
                f"--text_embedder_path={args.text_embedder_path}",
                f"--embed_batch_size={args.embed_batch_size}",
                f"--device=cuda:0",
                f"--num_shards={shard_count}",
                f"--shard_index={shard_index}",
            ]
            if args.table_names:
                command.append(f"--table_names={args.table_names}")
            if args.embed_chunk_size > 0:
                command.append(f"--embed_chunk_size={args.embed_chunk_size}")
            if args.force_cpu_aggregate:
                command.append("--force_cpu_aggregate")
            if args.download:
                command.append("--download")
            if args.copy_from_equivalent:
                command.append("--copy_from_equivalent")
            if args.skip_existing:
                command.append("--skip_existing")
            script_lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {target.remote_repo_root}",
                f"CUDA_VISIBLE_DEVICES={gpu} " + " ".join(command),
            ]
            windows.append(
                {
                    "name": window_name,
                    "gpu": gpu,
                    "shard_index": shard_index,
                    "num_shards": shard_count,
                    "parallelism_mode": args.parallelism_mode,
                    "remote_script": remote_script,
                    "remote_log": remote_log,
                    "script_lines": script_lines,
                }
            )

    return {
        "job_id": job_id,
        "kind": "graph_cache",
        "dataset": args.dataset,
        "cache_profile": args.cache_profile,
        "cache_dir": args.cache_dir,
        "parallelism_mode": args.parallelism_mode,
        "table_names": args.table_names,
        "target": target_name,
        "created_at": utc_stamp(),
        "status": "draft",
        "windows": windows,
    }


def build_gnn_repr_job(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    target_name = args.target or default_target_name(config)
    target = get_target_by_name(build_ssh_targets(config), target_name)
    slug = sanitize_slug(f"{args.dataset}_{args.job_suffix or 'gnn_repr'}")
    job_id = f"precompute_{slug}_{utc_stamp().lower()}"
    window_name = f"stage3-gnn-{sanitize_slug(args.dataset)}"
    remote_script = f"{target.remote_tmp_dir}/{window_name}.sh"
    remote_log = f"{target.remote_tmp_dir}/{window_name}.log"
    command = [
        target.remote_python,
        "-u",
        "-m",
        "gnn_repr",
        f"--dataset={args.dataset}",
        f"--text_embedder={args.text_embedder}",
        f"--train_steps={args.train_steps}",
        f"--batch_size={args.batch_size}",
        f"--num_neighbors={args.num_neighbors}",
        f"--channels={args.channels}",
        f"--num_layers={args.num_layers}",
        f"--aggr={args.aggr}",
        "--device=cuda:0",
        "--require_cache",
    ]
    if args.download:
        command.append("--download")
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {target.remote_repo_root}",
        f"CUDA_VISIBLE_DEVICES={args.gpu} " + " ".join(command),
    ]
    return {
        "job_id": job_id,
        "kind": "gnn_repr",
        "dataset": args.dataset,
        "target": target_name,
        "created_at": utc_stamp(),
        "status": "draft",
        "windows": [
            {
                "name": window_name,
                "gpu": args.gpu,
                "remote_script": remote_script,
                "remote_log": remote_log,
                "script_lines": script_lines,
            }
        ],
    }


def build_basis_job(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    target_name = args.target or default_target_name(config)
    target = get_target_by_name(build_ssh_targets(config), target_name)
    slug = sanitize_slug(f"{args.dataset}_{args.job_suffix or 'basis'}")
    job_id = f"precompute_{slug}_{utc_stamp().lower()}"
    window_name = f"stage3-basis-{sanitize_slug(args.dataset)}"
    remote_script = f"{target.remote_tmp_dir}/{window_name}.sh"
    remote_log = f"{target.remote_tmp_dir}/{window_name}.log"
    command = [
        target.remote_python,
        "-u",
        "-m",
        "basis",
        f"--dataset={args.dataset}",
        f"--model-type={args.model_type}",
        f"--db-scope={args.db_scope}",
        "--device=cuda:0",
    ]
    if args.download:
        command.append("--download")
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {target.remote_repo_root}",
        f"CUDA_VISIBLE_DEVICES={args.gpu} " + " ".join(command),
    ]
    return {
        "job_id": job_id,
        "kind": "basis",
        "dataset": args.dataset,
        "target": target_name,
        "created_at": utc_stamp(),
        "status": "draft",
        "windows": [
            {
                "name": window_name,
                "gpu": args.gpu,
                "remote_script": remote_script,
                "remote_log": remote_log,
                "script_lines": script_lines,
            }
        ],
    }


def summarize_job(job_path: Path, tail_lines: int) -> int:
    job = load_json(job_path)
    config = load_json(PIPELINE_CONFIG_PATH)
    target = get_target_by_name(build_ssh_targets(config), job["target"])
    windows = query_tmux_windows(target)
    any_active = False
    print(f"Job: {job_path}")
    print(f"Kind: {job['kind']}  Dataset: {job['dataset']}  Status: {job.get('status', 'unknown')}")
    for window in job["windows"]:
        present = window["name"] in windows
        any_active = any_active or present
        gpu_desc = window.get("gpu")
        if gpu_desc is None:
            gpu_desc = ",".join(str(gpu) for gpu in window.get("gpus", []))
        print(f"- Window {window['name']} gpu={gpu_desc} active={present} log={window['remote_log']}")
        tail_proc = ssh_command(
            target,
            f"tail -n {tail_lines} {shlex.quote(window['remote_log'])} 2>/dev/null || true",
        )
        tail_text = compact_log_tail(tail_proc.stdout, max_lines=tail_lines).strip()
        if tail_text:
            print(tail_text)
    if job.get("status") == "running" and not any_active:
        job["status"] = "completed_or_stopped"
        dump_json(job_path, job)
    return 0


def list_jobs() -> int:
    PRECOMPUTE_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(PRECOMPUTE_DIR.glob("*.json")):
        job = load_json(path)
        print(f"{path} :: {job.get('status', 'unknown')} :: {job.get('kind')} :: {job.get('dataset')}")
    return 0


def main() -> int:
    args = parse_args()
    config = load_json(PIPELINE_CONFIG_PATH)

    if args.command == "list-jobs":
        return list_jobs()

    if args.command == "status":
        return summarize_job(args.job, args.tail_lines)

    if args.command == "launch-cache":
        job = build_graph_cache_job(args, config)
    elif args.command == "launch-gnn-repr":
        job = build_gnn_repr_job(args, config)
    elif args.command == "launch-basis":
        job = build_basis_job(args, config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    job_path = precompute_job_path(job["job_id"])
    dump_json(job_path, job)
    launch_job(job)
    print(job_path)
    print(json.dumps(job, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
