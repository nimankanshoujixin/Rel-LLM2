from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
NOTES_DIR = ROOT / "stage3_notes"
BASELINE_REGISTRY_PATH = NOTES_DIR / "baseline_registry.json"
PIPELINE_CONFIG_PATH = NOTES_DIR / "pipeline_config.json"

EVAL_RE = re.compile(
    r"\[Eval\] Train step: (?P<step>\d+)/(?P<train_steps>\d+).*?"
    r"Val: (?P<val>\{.*?\}) \| "
    r"TestSubset: (?P<test>\{.*?\}) \| "
    r"SelectionSource: (?P<source>\w+) \| "
    r"Best selection metric: (?P<best>[-+]?\d*\.?\d+)",
)
BEST_SUBSET_RE = re.compile(r"Best TestSubset metrics: (?P<metrics>\{.*\})")
BEST_TEST_RE = re.compile(r"Best test metrics: (?P<metrics>\{.*\})")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sanitize_slug(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "_").replace("-", "_")
    lowered = re.sub(r"[^a-z0-9_]+", "", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    if not lowered:
        raise ValueError("Slug resolved to empty value.")
    return lowered


def exp_id(num: int) -> str:
    return f"EXP{num:03d}"


def exp_num(exp_name: str) -> int:
    match = re.fullmatch(r"EXP(\d{3})", exp_name)
    if not match:
        raise ValueError(f"Invalid experiment id: {exp_name}")
    return int(match.group(1))


@dataclass(frozen=True)
class SshTarget:
    name: str
    user: str
    host: str
    port: int
    remote_repo_root: str
    remote_python: str
    remote_tmp_dir: str
    tmux_session: str

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


def run_subprocess(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def ssh_command(target: SshTarget, remote_command: str) -> subprocess.CompletedProcess[str]:
    return run_subprocess(
        ["ssh", "-p", str(target.port), target.destination, remote_command],
        cwd=ROOT,
    )


def scp_to_remote(target: SshTarget, local_paths: list[Path], remote_dir: str) -> subprocess.CompletedProcess[str]:
    command = ["scp", "-P", str(target.port)]
    if any(path.is_dir() for path in local_paths):
        command.insert(1, "-r")
    command.extend(str(path) for path in local_paths)
    command.append(f"{target.destination}:{remote_dir}/")
    return run_subprocess(command, cwd=ROOT)


def scp_to_remote_path(
    target: SshTarget,
    local_paths: list[Path],
    remote_path: str,
) -> subprocess.CompletedProcess[str]:
    command = ["scp", "-P", str(target.port)]
    if any(path.is_dir() for path in local_paths):
        command.insert(1, "-r")
    command.extend(str(path) for path in local_paths)
    command.append(f"{target.destination}:{remote_path}")
    return run_subprocess(command, cwd=ROOT)


def scp_from_remote(target: SshTarget, remote_paths: list[str], local_dir: Path) -> subprocess.CompletedProcess[str]:
    local_dir.mkdir(parents=True, exist_ok=True)
    command = ["scp", "-P", str(target.port)]
    command.extend(f"{target.destination}:{remote_path}" for remote_path in remote_paths)
    command.append(str(local_dir))
    return run_subprocess(command, cwd=ROOT)


def build_ssh_targets(config: dict[str, Any]) -> list[SshTarget]:
    targets_cfg = config.get("remote_targets")
    if targets_cfg:
        targets = []
        for index, payload in enumerate(targets_cfg):
            name = payload.get("name", f"target{index}")
            targets.append(
                SshTarget(
                    name=name,
                    user=payload["user"],
                    host=payload["host"],
                    port=int(payload["port"]),
                    remote_repo_root=payload.get("remote_repo_root", config["remote_repo_root"]),
                    remote_python=payload.get("remote_python", config["remote_python"]),
                    remote_tmp_dir=payload.get("remote_tmp_dir", config["remote_tmp_dir"]),
                    tmux_session=payload.get("tmux_session", config["tmux_session"]),
                )
            )
        return targets

    remote = config["remote_ssh"]
    return [
        SshTarget(
            name=remote.get("name", "default"),
            user=remote["user"],
            host=remote["host"],
            port=int(remote["port"]),
            remote_repo_root=config["remote_repo_root"],
            remote_python=config["remote_python"],
            remote_tmp_dir=config["remote_tmp_dir"],
            tmux_session=config["tmux_session"],
        )
    ]


def get_target_by_name(targets: list[SshTarget], name: str) -> SshTarget:
    for target in targets:
        if target.name == name:
            return target
    raise KeyError(f"Unknown remote target: {name}")


def parse_tmux_windows(stdout: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        index_part, rest = line.split(":", 1)
        window_name = rest.split()[0]
        active = "*" in line
        result[window_name] = {
            "index": index_part.strip(),
            "raw": line,
            "active": active,
        }
    return result


def query_tmux_windows(target: SshTarget) -> dict[str, dict[str, Any]]:
    proc = ssh_command(target, f"tmux list-windows -t {target.tmux_session} 2>/dev/null || true")
    return parse_tmux_windows(proc.stdout)


def parse_gpu_table(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) not in {3, 4}:
            continue
        if len(parts) == 4:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "utilization_gpu": int(parts[3]),
                }
            )
        else:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "utilization_gpu": int(parts[2]),
                }
            )
    return rows


def query_gpus(target: SshTarget) -> list[dict[str, Any]]:
    proc = ssh_command(
        target,
        "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits",
    )
    return parse_gpu_table(proc.stdout)


def parse_compute_app_table(stdout: str) -> list[dict[str, Any]]:
    rows = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        pid_text = parts[1]
        try:
            pid = int(pid_text)
        except ValueError:
            pid = -1
        used_memory_text = parts[3]
        try:
            used_memory_mb = int(used_memory_text)
        except ValueError:
            used_memory_mb = 0
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": pid,
                "process_name": parts[2],
                "used_memory_mb": used_memory_mb,
            }
        )
    return rows


def query_compute_apps(target: SshTarget) -> list[dict[str, Any]]:
    proc = ssh_command(
        target,
        "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
    )
    return parse_compute_app_table(proc.stdout)


def query_live_pids(target: SshTarget, pids: list[int]) -> set[int]:
    if not pids:
        return set()
    pid_spec = ",".join(str(pid) for pid in sorted(set(pids)))
    proc = ssh_command(target, f"ps -o pid= -p {pid_spec} 2>/dev/null || true")
    live_pids: set[int] = set()
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            live_pids.add(int(line))
        except ValueError:
            continue
    return live_pids


def annotate_compute_apps_with_liveness(
    target: SshTarget,
    compute_apps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pids = [app["pid"] for app in compute_apps if int(app["pid"]) > 0]
    live_pids = query_live_pids(target, pids)
    annotated = []
    for app in compute_apps:
        annotated_app = dict(app)
        annotated_app["process_live"] = (
            annotated_app["pid"] in live_pids
            and annotated_app["process_name"] != "[Not Found]"
        )
        annotated.append(annotated_app)
    return annotated


def group_compute_apps_by_gpu_uuid(
    compute_apps: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for app in compute_apps:
        grouped.setdefault(str(app["gpu_uuid"]), []).append(app)
    return grouped


def gpu_idle_thresholds(config: dict[str, Any]) -> tuple[int, int]:
    return (
        int(config.get("gpu_idle_max_memory_mb", 2048)),
        int(config.get("gpu_idle_max_utilization", 10)),
    )


def residual_gpu_idle_memory_threshold(config: dict[str, Any]) -> int:
    base_threshold, _ = gpu_idle_thresholds(config)
    return int(config.get("gpu_idle_residual_max_memory_mb", base_threshold))


def classify_gpu_idle(
    gpu_row: dict[str, Any],
    config: dict[str, Any],
    compute_apps_by_gpu_uuid: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[bool, str]:
    max_mem_mb, max_util = gpu_idle_thresholds(config)
    memory_used_mb = int(gpu_row["memory_used_mb"])
    utilization_gpu = int(gpu_row["utilization_gpu"])
    if (
        memory_used_mb <= max_mem_mb
        and utilization_gpu <= max_util
    ):
        return True, "strict"

    residual_max_mem_mb = residual_gpu_idle_memory_threshold(config)
    if (
        residual_max_mem_mb > max_mem_mb
        and utilization_gpu <= max_util
        and memory_used_mb <= residual_max_mem_mb
    ):
        gpu_uuid = str(gpu_row.get("uuid", ""))
        gpu_apps = (compute_apps_by_gpu_uuid or {}).get(gpu_uuid, [])
        if gpu_apps and all(not bool(app.get("process_live", False)) for app in gpu_apps):
            return True, "residual_stale_compute_app"

    return False, "busy"


def is_gpu_idle(
    gpu_row: dict[str, Any],
    config: dict[str, Any],
    compute_apps_by_gpu_uuid: dict[str, list[dict[str, Any]]] | None = None,
) -> bool:
    idle, _ = classify_gpu_idle(gpu_row, config, compute_apps_by_gpu_uuid)
    return idle


def annotate_gpu_idle_states(
    gpus: list[dict[str, Any]],
    config: dict[str, Any],
    compute_apps_by_gpu_uuid: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotated_gpus: list[dict[str, Any]] = []
    idle_gpus: list[dict[str, Any]] = []
    for gpu in gpus:
        annotated_gpu = dict(gpu)
        idle, idle_reason = classify_gpu_idle(
            annotated_gpu,
            config,
            compute_apps_by_gpu_uuid,
        )
        annotated_gpu["idle"] = idle
        annotated_gpu["idle_reason"] = idle_reason
        annotated_gpus.append(annotated_gpu)
        if idle:
            idle_gpus.append(annotated_gpu)
    return annotated_gpus, idle_gpus


def probe_remote_targets(targets: list[SshTarget], config: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for target in targets:
        tmux_proc = ssh_command(target, f"tmux list-windows -t {target.tmux_session} 2>/dev/null || true")
        gpu_proc = ssh_command(
            target,
            "nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits",
        )
        apps_proc = ssh_command(
            target,
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true",
        )
        if tmux_proc.returncode != 0 and gpu_proc.returncode != 0:
            snapshot[target.name] = {
                "target": target,
                "reachable": False,
                "windows": {},
                "gpus": [],
                "idle_gpus": [],
                "compute_apps": [],
                "errors": [tmux_proc.stderr.strip(), gpu_proc.stderr.strip()],
            }
            continue
        gpus = parse_gpu_table(gpu_proc.stdout)
        compute_apps = annotate_compute_apps_with_liveness(
            target,
            parse_compute_app_table(apps_proc.stdout),
        )
        compute_apps_by_gpu_uuid = group_compute_apps_by_gpu_uuid(compute_apps)
        annotated_gpus, idle_gpus = annotate_gpu_idle_states(
            gpus,
            config,
            compute_apps_by_gpu_uuid,
        )
        snapshot[target.name] = {
            "target": target,
            "reachable": True,
            "windows": parse_tmux_windows(tmux_proc.stdout),
            "gpus": annotated_gpus,
            "idle_gpus": idle_gpus,
            "compute_apps": compute_apps,
            "errors": [
                error
                for error in (tmux_proc.stderr.strip(), gpu_proc.stderr.strip(), apps_proc.stderr.strip())
                if error
            ],
        }
    return snapshot


def report_path_for_candidate(candidate_path: Path) -> Path:
    report_dir = ROOT / load_json(PIPELINE_CONFIG_PATH)["report_dir"]
    return report_dir / f"{candidate_path.stem}.report.json"


def finalized_candidate_status(global_verdict: str) -> str:
    if global_verdict == "promotable":
        return "promotable"
    if global_verdict == "retune_plausible":
        return "retune_plausible"
    if global_verdict == "failed":
        return "failed"
    return "stopped"


def update_candidate_status_from_report(candidate_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    candidate = load_json(candidate_path)
    candidate["status"] = finalized_candidate_status(
        str(report.get("candidate_status", report["global_verdict"]))
    )
    dump_json(candidate_path, candidate)
    return candidate


def format_shell_value(value: Any) -> str:
    if isinstance(value, bool):
        raise TypeError("Boolean values must be handled separately.")
    if isinstance(value, str):
        return value
    return str(value)


def build_flag_lines(args_map: OrderedDict[str, Any]) -> list[str]:
    lines: list[str] = []
    items = list(args_map.items())
    for index, (key, value) in enumerate(items):
        suffix = " \\" if index < len(items) - 1 else ""
        flag = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            lines.append(f"  {flag}{suffix}")
        elif isinstance(value, list):
            joined = " ".join(format_shell_value(item) for item in value)
            lines.append(f"  {flag} {joined}{suffix}")
        else:
            lines.append(f"  {flag}={format_shell_value(value)}{suffix}")
    return lines


def merge_args(
    baseline_args: dict[str, Any],
    common_overrides: dict[str, Any],
    task_overrides: dict[str, Any],
) -> OrderedDict[str, Any]:
    merged: OrderedDict[str, Any] = OrderedDict()
    for key, value in baseline_args.items():
        merged[key] = value
    for override_map in (common_overrides, task_overrides):
        for key, value in override_map.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
    return merged


def candidate_launch_mode(candidate: dict[str, Any]) -> str:
    return str(candidate.get("launch_mode", "parallel"))


def normalize_gpu_spec(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def compact_gpu_spec(gpus: list[int]) -> int | list[int]:
    if len(gpus) == 1:
        return gpus[0]
    return gpus


def format_gpu_spec(value: Any) -> str:
    gpus = normalize_gpu_spec(value)
    if not gpus:
        return "None"
    return ",".join(str(gpu) for gpu in gpus)


def merged_task_args(
    candidate: dict[str, Any],
    baselines: dict[str, Any],
    task_name: str,
) -> OrderedDict[str, Any]:
    return merge_args(
        baselines[task_name]["args"],
        candidate.get("common_overrides", {}),
        candidate.get("task_overrides", {}).get(task_name, {}),
    )


def batch_equivalent_world_size(
    global_batch_size: int,
    max_gpus: int,
) -> int:
    if global_batch_size < 1:
        raise ValueError(f"Invalid batch size for batch-equivalent launch: {global_batch_size}")
    capped = max(1, min(global_batch_size, max_gpus))
    for divisor in range(capped, 0, -1):
        if global_batch_size % divisor == 0:
            return divisor
    return 1


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    if world_size < 1:
        raise ValueError(f"Invalid world size: {world_size}")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Cannot preserve batch equivalence when batch_size={global_batch_size} and world_size={world_size}."
        )
    return max(1, global_batch_size // world_size)


def batch_equivalent_world_size_options(global_batch_size: int, max_gpus: int) -> list[int]:
    if global_batch_size < 1:
        raise ValueError(f"Invalid batch size for batch-equivalent launch: {global_batch_size}")
    capped = max(1, min(global_batch_size, max_gpus))
    return [
        world_size
        for world_size in range(capped, 0, -1)
        if global_batch_size % world_size == 0
    ]


def excluded_gpu_indices(candidate: dict[str, Any], config: dict[str, Any]) -> set[int]:
    excluded = candidate.get("gpu_exclude", config.get("gpu_exclude", []))
    return {int(gpu) for gpu in normalize_gpu_spec(excluded)}


def filtered_idle_gpu_rows(
    snapshot: dict[str, Any],
    target_name: str,
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    excluded = excluded_gpu_indices(candidate, config)
    return [
        row
        for row in snapshot[target_name]["idle_gpus"]
        if int(row["index"]) not in excluded
    ]


def estimate_packed_wave_plan(
    task_order: list[str],
    batch_sizes: dict[str, int],
    task_cost_weights: dict[str, float],
    gpu_count: int,
) -> list[dict[str, int]]:
    if gpu_count < 1:
        raise ValueError("Packed scheduling requires at least one GPU.")

    memo: dict[tuple[str, ...], tuple[float, int, float, list[dict[str, int]]]] = {}
    task_rank = {task: index for index, task in enumerate(task_order)}
    max_world_sizes = {
        task: batch_equivalent_world_size(batch_sizes[task], gpu_count)
        for task in task_order
    }

    def search(remaining: tuple[str, ...]) -> tuple[float, int, float, list[dict[str, int]]]:
        if not remaining:
            return 0.0, 0, 0.0, []
        if remaining in memo:
            return memo[remaining]

        best: tuple[float, int, float, list[dict[str, int]]] | None = None
        remaining_set = set(remaining)
        for subset_mask in range(1, 1 << len(remaining)):
            wave_tasks = tuple(
                task
                for index, task in enumerate(remaining)
                if subset_mask & (1 << index)
            )
            wave_gpu_need = sum(max_world_sizes[task] for task in wave_tasks)
            if wave_gpu_need > gpu_count:
                continue

            current = {task: max_world_sizes[task] for task in wave_tasks}
            wave_time = max(
                task_cost_weights.get(task, 1.0) / float(current[task])
                for task in wave_tasks
            )
            next_remaining = tuple(
                task
                for task in task_order
                if task in remaining_set and task not in current
            )
            rest_time, rest_wave_count, rest_gpu_time, rest_plan = search(next_remaining)
            total_time = wave_time + rest_time
            wave_count = 1 + rest_wave_count
            gpu_time = float(wave_gpu_need) * wave_time + rest_gpu_time
            ordered_current = {
                task: current[task]
                for task in sorted(current, key=lambda item: task_rank[item])
            }
            candidate_plan = [ordered_current, *rest_plan]
            score = (total_time, wave_count, -gpu_time)
            if best is None or score < (best[0], best[1], -best[2]):
                best = (total_time, wave_count, gpu_time, candidate_plan)

        if best is None:
            raise RuntimeError(
                f"Could not build a packed batch-equivalent plan for tasks={list(remaining)} "
                f"with gpu_count={gpu_count}."
            )
        memo[remaining] = best
        return best

    return search(tuple(task_order))[3]


def ensure_candidate_valid(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    task_order = config["task_order"]
    run_ids = candidate.get("task_run_ids", {})
    missing_run_ids = [task for task in task_order if task not in run_ids]
    if missing_run_ids:
        raise ValueError(f"Candidate missing task_run_ids for: {missing_run_ids}")

    literature = candidate.get("literature", {})
    papers = literature.get("papers", [])
    queries = literature.get("queries", [])
    prior_experiments = candidate.get("evidence", {}).get("prior_experiments", [])
    source_type = candidate.get("source_type", "")

    if source_type in {"paper", "paper+ablation"} and not papers:
        raise ValueError("Paper-backed candidates must include at least one paper.")
    if not papers and not prior_experiments:
        raise ValueError(
            "Candidate must include either literature papers or prior experiment evidence."
        )
    if not queries:
        raise ValueError("Candidate must include literature search queries.")


def candidate_filename(start_exp: int, slug: str) -> str:
    return f"exp{start_exp:03d}_{slug}.json"


def make_candidate_template(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    task_order = config["task_order"]
    run_ids = {
        task_name: exp_id(args.start_exp + index)
        for index, task_name in enumerate(task_order)
    }
    return {
        "bundle_id": f"BUNDLE-{args.start_exp:03d}",
        "slug": sanitize_slug(args.slug),
        "title": args.title,
        "family": args.family,
        "source_type": args.source_type,
        "launch_mode": "parallel",
        "status": "draft",
        "task_run_ids": run_ids,
        "literature": {
            "queries": [
                "",
                ""
            ],
            "papers": [
                {
                    "title": "",
                    "url": "",
                    "claim": "",
                    "relevance": "",
                    "status": "todo"
                }
            ]
        },
        "evidence": {
            "prior_experiments": [],
            "why_now": ""
        },
        "hypothesis": "",
        "common_overrides": {},
        "task_overrides": {
            task_name: {}
            for task_name in task_order
        },
        "notes": []
    }


def resolve_task_launches(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    launch_mode = candidate_launch_mode(candidate)
    if launch_mode in {"serial_batch_equivalent", "packed_batch_equivalent"}:
        raise ValueError(
            f"{launch_mode} launch plans must be re-resolved from current idle GPUs. "
            "Pass an explicit task_launches plan from assign_candidate_launches(...) instead of "
            "reusing the recorded task_launches snapshot."
        )
    if launch_mode != "parallel":
        raise ValueError(f"Unsupported launch_mode: {launch_mode}")

    targets = build_ssh_targets(config)
    default_target = targets[0]
    explicit_launches = candidate.get("task_launches", {})
    task_targets = candidate.get("task_targets", {})
    task_gpus = candidate.get("task_gpus", {})
    task_num_gpus = candidate.get("task_num_gpus", {})

    launches: dict[str, dict[str, Any]] = {}
    for task_name in config["task_order"]:
        if task_name in explicit_launches:
            payload = explicit_launches[task_name]
            target = get_target_by_name(targets, payload["target"])
            gpu_spec = compact_gpu_spec(normalize_gpu_spec(payload["gpu"]))
            launches[task_name] = {
                "target": target.name,
                "gpu": gpu_spec,
                "window_name": payload.get("window_name", f"stage3-exp{exp_num(candidate['task_run_ids'][task_name]):03d}"),
            }
            continue

        target_name = task_targets.get(task_name, default_target.name)
        target = get_target_by_name(targets, target_name)
        requested_gpus = normalize_gpu_spec(task_gpus.get(task_name))
        if not requested_gpus:
            requested_count = int(task_num_gpus.get(task_name, 1))
            if requested_count <= 1:
                requested_gpus = [int(load_json(BASELINE_REGISTRY_PATH)["tasks"][task_name]["default_gpu"])]
            else:
                raise ValueError(
                    f"Candidate requests {requested_count} GPUs for {task_name} but has no resolved task_launches yet."
                )
        launches[task_name] = {
            "target": target.name,
            "gpu": compact_gpu_spec(requested_gpus),
            "window_name": f"stage3-exp{exp_num(candidate['task_run_ids'][task_name]):03d}",
        }
    return launches


def assign_serial_batch_equivalent_launches(
    candidate_path: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate = load_json(candidate_path)
    targets = build_ssh_targets(config)
    baselines = load_json(BASELINE_REGISTRY_PATH)["tasks"]
    snapshot = probe_remote_targets(targets, config)
    explicit_target = candidate.get("serial_target")
    serial_gpu_cap = candidate.get("serial_gpu_cap")
    task_gpus = candidate.get("task_gpus", {})

    reachable_targets = []
    for target in targets:
        state = snapshot[target.name]
        if not state["reachable"]:
            continue
        if explicit_target is not None and target.name != explicit_target:
            continue
        reachable_targets.append(target)
    if not reachable_targets:
        raise RuntimeError(
            f"No reachable target available for serial_batch_equivalent launch (serial_target={explicit_target!r})."
        )

    target = max(
        reachable_targets,
        key=lambda item: len(snapshot[item.name]["idle_gpus"]),
    )
    idle_gpu_rows = sorted(
        filtered_idle_gpu_rows(snapshot, target.name, candidate, config),
        key=lambda row: int(row["index"]),
    )
    if not idle_gpu_rows:
        raise RuntimeError(f"No idle GPUs available on target {target.name} for serial_batch_equivalent launch.")

    idle_gpu_count = len(idle_gpu_rows)
    if serial_gpu_cap is not None:
        idle_gpu_count = min(idle_gpu_count, int(serial_gpu_cap))
    if idle_gpu_count < 1:
        raise RuntimeError(
            f"serial_batch_equivalent launch on {target.name} resolved to zero usable GPUs after serial_gpu_cap."
        )

    launches: dict[str, dict[str, Any]] = {}
    for task_name in config["task_order"]:
        merged_args = merged_task_args(candidate, baselines, task_name)
        global_batch_size = int(merged_args["batch_size"])
        explicit_gpu_list = normalize_gpu_spec(task_gpus.get(task_name))
        if explicit_gpu_list:
            available_gpu_indices = {int(row["index"]) for row in idle_gpu_rows}
            missing = [gpu for gpu in explicit_gpu_list if gpu not in available_gpu_indices]
            if missing:
                raise RuntimeError(
                    f"serial_batch_equivalent launch requested unavailable GPUs for {task_name}: {missing}"
                )
            world_size = len(explicit_gpu_list)
            if world_size < 1 or global_batch_size % world_size != 0:
                raise RuntimeError(
                    f"serial_batch_equivalent launch for {task_name} requires a GPU count dividing batch_size={global_batch_size}; "
                    f"requested GPUs={explicit_gpu_list}"
                )
            gpu_list = explicit_gpu_list
        else:
            world_size = batch_equivalent_world_size(global_batch_size, idle_gpu_count)
            gpu_list = [int(row["index"]) for row in idle_gpu_rows[:world_size]]
        launches[task_name] = {
            "target": target.name,
            "gpu": compact_gpu_spec(gpu_list),
            "window_name": f"stage3-exp{exp_num(candidate['task_run_ids'][task_name]):03d}",
            "baseline_default_gpu": baselines[task_name]["default_gpu"],
            "per_rank_batch_size": per_rank_batch_size(global_batch_size, world_size),
            "launch_world_size": world_size,
            "launch_mode": "serial_batch_equivalent",
        }
    return launches


def assign_packed_batch_equivalent_launches(
    candidate_path: Path,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidate = load_json(candidate_path)
    targets = build_ssh_targets(config)
    baselines = load_json(BASELINE_REGISTRY_PATH)["tasks"]
    snapshot = probe_remote_targets(targets, config)
    explicit_target = candidate.get("packed_target", candidate.get("serial_target"))
    packed_gpu_cap = candidate.get("packed_gpu_cap", candidate.get("serial_gpu_cap"))

    reachable_targets = []
    for target in targets:
        state = snapshot[target.name]
        if not state["reachable"]:
            continue
        if explicit_target is not None and target.name != explicit_target:
            continue
        reachable_targets.append(target)
    if not reachable_targets:
        raise RuntimeError(
            f"No reachable target available for packed_batch_equivalent launch (packed_target={explicit_target!r})."
        )

    target = max(
        reachable_targets,
        key=lambda item: len(filtered_idle_gpu_rows(snapshot, item.name, candidate, config)),
    )
    idle_gpu_rows = sorted(
        filtered_idle_gpu_rows(snapshot, target.name, candidate, config),
        key=lambda row: int(row["index"]),
    )
    if packed_gpu_cap is not None:
        idle_gpu_rows = idle_gpu_rows[: int(packed_gpu_cap)]
    if not idle_gpu_rows:
        raise RuntimeError(f"No idle GPUs available on target {target.name} for packed_batch_equivalent launch.")

    idle_gpu_indices = [int(row["index"]) for row in idle_gpu_rows]
    task_order = list(config["task_order"])
    batch_sizes = {
        task_name: int(merged_task_args(candidate, baselines, task_name)["batch_size"])
        for task_name in task_order
    }
    task_cost_weights = {
        str(task_name): float(weight)
        for task_name, weight in candidate.get("task_cost_weights", {}).items()
    }
    wave_plan = estimate_packed_wave_plan(
        task_order,
        batch_sizes,
        task_cost_weights,
        len(idle_gpu_indices),
    )

    launches: dict[str, dict[str, Any]] = {}
    for wave_index, wave in enumerate(wave_plan):
        cursor = 0
        for task_name, world_size in wave.items():
            gpu_list = idle_gpu_indices[cursor : cursor + world_size]
            cursor += world_size
            global_batch_size = batch_sizes[task_name]
            launches[task_name] = {
                "target": target.name,
                "gpu": compact_gpu_spec(gpu_list),
                "window_name": f"stage3-exp{exp_num(candidate['task_run_ids'][task_name]):03d}",
                "baseline_default_gpu": baselines[task_name]["default_gpu"],
                "per_rank_batch_size": per_rank_batch_size(global_batch_size, world_size),
                "launch_world_size": world_size,
                "launch_mode": "packed_batch_equivalent",
                "launch_wave": wave_index,
            }
    return launches


def assign_candidate_launches(candidate_path: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidate = load_json(candidate_path)
    launch_mode = candidate_launch_mode(candidate)
    if launch_mode == "serial_batch_equivalent":
        return assign_serial_batch_equivalent_launches(candidate_path, config)
    if launch_mode == "packed_batch_equivalent":
        return assign_packed_batch_equivalent_launches(candidate_path, config)
    if launch_mode != "parallel":
        raise ValueError(f"Unsupported launch_mode: {launch_mode}")

    targets = build_ssh_targets(config)
    baseline_registry = load_json(BASELINE_REGISTRY_PATH)["tasks"]
    snapshot = probe_remote_targets(targets, config)
    pool: list[tuple[str, int]] = []

    for target in targets:
        state = snapshot[target.name]
        if not state["reachable"]:
            continue
        for gpu in filtered_idle_gpu_rows(snapshot, target.name, candidate, config):
            pool.append((target.name, int(gpu["index"])))

    reserved = {
        (payload["target"], gpu_index)
        for payload in candidate.get("task_launches", {}).values()
        for gpu_index in normalize_gpu_spec(payload.get("gpu"))
    }
    available_pool = [entry for entry in sorted(pool, key=lambda item: (item[0], item[1])) if entry not in reserved]
    task_targets = candidate.get("task_targets", {})
    task_gpus = candidate.get("task_gpus", {})
    task_num_gpus = candidate.get("task_num_gpus", {})
    launches: dict[str, dict[str, Any]] = {}

    for task_name in config["task_order"]:
        explicit_target = task_targets.get(task_name)
        explicit_gpu = task_gpus.get(task_name)
        explicit_gpu_list = normalize_gpu_spec(explicit_gpu)
        requested_count = len(explicit_gpu_list) if explicit_gpu_list else int(task_num_gpus.get(task_name, 1))
        run_id = candidate["task_run_ids"][task_name]
        window_name = f"stage3-exp{exp_num(run_id):03d}"

        if explicit_target is not None and explicit_gpu_list:
            launches[task_name] = {
                "target": explicit_target,
                "gpu": compact_gpu_spec(explicit_gpu_list),
                "window_name": window_name,
            }
            continue

        grouped_candidates: dict[str, list[tuple[int, int]]] = {}
        for index, (target_name, gpu_index) in enumerate(available_pool):
            if explicit_target is not None and target_name != explicit_target:
                continue
            if explicit_gpu_list and gpu_index not in explicit_gpu_list:
                continue
            grouped_candidates.setdefault(target_name, []).append((index, gpu_index))

        selected_target = None
        selected_entries: list[tuple[int, int]] = []
        for target_name in sorted(grouped_candidates):
            entries = grouped_candidates[target_name]
            if explicit_gpu_list:
                desired = [gpu for gpu in explicit_gpu_list if any(candidate_gpu == gpu for _, candidate_gpu in entries)]
                if len(desired) != len(explicit_gpu_list):
                    continue
                entry_map = {gpu: index for index, gpu in entries}
                selected_target = target_name
                selected_entries = [(entry_map[gpu], gpu) for gpu in explicit_gpu_list]
                break
            if len(entries) >= requested_count:
                selected_target = target_name
                selected_entries = entries[:requested_count]
                break

        if selected_target is None:
            raise RuntimeError(
                f"Not enough idle GPUs to schedule {task_name}. "
                f"Requested target={explicit_target!r} gpu={explicit_gpu!r} count={requested_count!r}."
            )

        used_indices = sorted((index for index, _ in selected_entries), reverse=True)
        selected_gpu_list = [gpu for _, gpu in selected_entries]
        for index in used_indices:
            available_pool.pop(index)
        launches[task_name] = {
            "target": selected_target,
            "gpu": compact_gpu_spec(selected_gpu_list),
            "window_name": window_name,
            "baseline_default_gpu": baseline_registry[task_name]["default_gpu"],
        }

    return launches


def render_candidate(
    candidate_path: Path,
    allow_no_papers: bool = False,
    task_launches: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baselines = load_json(BASELINE_REGISTRY_PATH)["tasks"]
    config = load_json(PIPELINE_CONFIG_PATH)
    candidate = load_json(candidate_path)
    launch_mode = candidate_launch_mode(candidate)
    if allow_no_papers:
        candidate.setdefault("literature", {}).setdefault("queries", ["manual override"])
    ensure_candidate_valid(candidate, config)

    wrapper_output_dir = ROOT / config["wrapper_output_dir"]
    wrapper_output_dir.mkdir(parents=True, exist_ok=True)

    slug = sanitize_slug(candidate["slug"])
    task_order = config["task_order"]
    targets = build_ssh_targets(config)
    if task_launches is not None:
        launch_plan = task_launches
    elif launch_mode in {"serial_batch_equivalent", "packed_batch_equivalent"}:
        # For batch-equivalent launch modes, recorded task_launches are runtime snapshots only.
        # Always re-resolve placement from the current idle GPU pool unless the caller has
        # already provided an explicit launch plan for this render.
        launch_plan = assign_candidate_launches(candidate_path, config)
    else:
        launch_plan = resolve_task_launches(candidate, config)

    rendered: dict[str, Any] = {
        "candidate": str(candidate_path),
        "launch_mode": launch_mode,
        "wrappers": {},
        "launchers": {},
    }
    launcher_lines_by_target: dict[str, list[str]] = {}

    for task_name in task_order:
        baseline = baselines[task_name]
        run_id = candidate["task_run_ids"][task_name]
        run_num = exp_num(run_id)
        launch_payload = launch_plan[task_name]
        target = get_target_by_name(targets, launch_payload["target"])
        script_name = (
            f"{run_id.lower()}_{slug}_with_test_subset_selection_"
            f"{task_name.replace('-', '_')}.sh"
        )
        log_name = f"stage3-exp{run_num:03d}.log"
        window_name = launch_payload.get("window_name", f"stage3-exp{run_num:03d}")
        gpu_list = normalize_gpu_spec(launch_payload["gpu"])
        if not gpu_list:
            raise ValueError(f"Launch payload for {task_name} has no GPUs: {launch_payload}")
        gpu_spec = ",".join(str(gpu) for gpu in gpu_list)
        master_port = 29500 + run_num

        merged_args = merged_task_args(candidate, baselines, task_name)
        if "per_rank_batch_size" in launch_payload:
            merged_args["batch_size"] = int(launch_payload["per_rank_batch_size"])
        flag_lines = build_flag_lines(merged_args)

        if len(gpu_list) == 1:
            launch_command = f"CUDA_VISIBLE_DEVICES={gpu_spec} {target.remote_python} -u main.py \\"
        else:
            launch_command = (
                f"CUDA_VISIBLE_DEVICES={gpu_spec} OMP_NUM_THREADS=1 "
                f"{target.remote_python} -m torch.distributed.run --nproc_per_node={len(gpu_list)} "
                f"--master_port={master_port} main.py \\"
            )

        script_text = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -eo pipefail",
                "",
                f"cd {target.remote_repo_root}",
                "",
                launch_command,
                *flag_lines,
                "",
            ]
        )
        script_path = wrapper_output_dir / script_name
        script_path.write_text(script_text, encoding="utf-8", newline="\n")

        rendered["wrappers"][task_name] = {
            "run_id": run_id,
            "target": target.name,
            "gpu": compact_gpu_spec(gpu_list),
            "script": str(script_path),
            "remote_script": f"{target.remote_tmp_dir}/{script_name}",
            "remote_log": f"{target.remote_tmp_dir}/{log_name}",
            "window_name": window_name,
            "per_rank_batch_size": launch_payload.get("per_rank_batch_size"),
            "launch_world_size": launch_payload.get("launch_world_size", len(gpu_list)),
            "launch_wave": launch_payload.get("launch_wave"),
        }
        if launch_mode == "parallel":
            launcher_lines = launcher_lines_by_target.setdefault(
                target.name,
                [
                    "#!/usr/bin/env bash",
                    "set -euo pipefail",
                    "",
                ],
            )
            launcher_lines.extend(
                [
                    f"tmux has-session -t {target.tmux_session} >/dev/null 2>&1 || tmux new-session -d -s {target.tmux_session}",
                    f"tmux kill-window -t {target.tmux_session}:{window_name} >/dev/null 2>&1 || true",
                    f": > {target.remote_tmp_dir}/{log_name}",
                    (
                        f"tmux new-window -d -t {target.tmux_session}: -n {window_name} "
                        f"\"sh -lc 'bash {target.remote_tmp_dir}/{script_name} > "
                        f"{target.remote_tmp_dir}/{log_name} 2>&1'\""
                    ),
                    "",
                ]
            )

    if launch_mode == "parallel":
        for target_name, launcher_lines in launcher_lines_by_target.items():
            target = get_target_by_name(targets, target_name)
            launcher_name = (
                f"launch_{candidate['task_run_ids'][task_order[0]].lower()}_{slug}_{target.name}_tmux.sh"
            )
            launcher_lines.append(f"tmux list-windows -t {target.tmux_session}")
            launcher_path = wrapper_output_dir / launcher_name
            launcher_path.write_text("\n".join(launcher_lines) + "\n", encoding="utf-8", newline="\n")
            rendered["launchers"][target_name] = {
                "script": str(launcher_path),
                "mode": "parallel",
            }
    elif launch_mode in {"serial_batch_equivalent", "packed_batch_equivalent"}:
        target_names = sorted({payload["target"] for payload in rendered["wrappers"].values()})
        if len(target_names) != 1:
            raise ValueError(
                f"{launch_mode} currently requires exactly one target, got {target_names}."
            )
        target_name = target_names[0]
        target = get_target_by_name(targets, target_name)
        launcher_suffix = "packed" if launch_mode == "packed_batch_equivalent" else "serial"
        launcher_name = (
            f"launch_{candidate['task_run_ids'][task_order[0]].lower()}_{slug}_{target.name}_{launcher_suffix}_tmux.sh"
        )
        controller_window = f"stage3-bundle-exp{exp_num(candidate['task_run_ids'][task_order[0]]):03d}-{launcher_suffix}"
        controller_log = f"{target.remote_tmp_dir}/stage3-bundle-exp{exp_num(candidate['task_run_ids'][task_order[0]]):03d}-{launcher_suffix}.log"
        launcher_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f"tmux has-session -t {target.tmux_session} >/dev/null 2>&1 || tmux new-session -d -s {target.tmux_session}",
            "",
        ]
        if launch_mode == "serial_batch_equivalent":
            for task_name in task_order:
                payload = rendered["wrappers"][task_name]
                quoted_window = shlex.quote(payload["window_name"])
                launcher_lines.extend(
                    [
                        f"tmux kill-window -t {target.tmux_session}:{payload['window_name']} >/dev/null 2>&1 || true",
                        f": > {payload['remote_log']}",
                        (
                            f"tmux new-window -d -t {target.tmux_session}: -n {payload['window_name']} "
                            f"\"sh -lc 'bash {payload['remote_script']} > {payload['remote_log']} 2>&1'\""
                        ),
                        (
                            f"while tmux list-windows -t {target.tmux_session} -F '#W' | "
                            f"grep -Fxq -- {quoted_window}; do sleep 30; done"
                        ),
                        "",
                    ]
                )
        else:
            max_wave = max(int(payload.get("launch_wave", 0)) for payload in rendered["wrappers"].values())
            for wave_index in range(max_wave + 1):
                wave_payloads = [
                    rendered["wrappers"][task_name]
                    for task_name in task_order
                    if int(rendered["wrappers"][task_name].get("launch_wave", 0)) == wave_index
                ]
                launcher_lines.append(f"# wave {wave_index}")
                for payload in wave_payloads:
                    launcher_lines.extend(
                        [
                            f"tmux kill-window -t {target.tmux_session}:{payload['window_name']} >/dev/null 2>&1 || true",
                            f": > {payload['remote_log']}",
                            (
                                f"tmux new-window -d -t {target.tmux_session}: -n {payload['window_name']} "
                                f"\"sh -lc 'bash {payload['remote_script']} > {payload['remote_log']} 2>&1'\""
                            ),
                        ]
                    )
                quoted_windows = " ".join(shlex.quote(payload["window_name"]) for payload in wave_payloads)
                launcher_lines.extend(
                    [
                        f"wave_windows=( {quoted_windows} )",
                        "while true; do",
                        "  active=0",
                        "  for wave_window in \"${wave_windows[@]}\"; do",
                        f"    if tmux list-windows -t {target.tmux_session} -F '#W' | grep -Fxq -- \"$wave_window\"; then",
                        "      active=1",
                        "      break",
                        "    fi",
                        "  done",
                        "  if [ \"$active\" -eq 0 ]; then break; fi",
                        "  sleep 30",
                        "done",
                        "",
                    ]
                )
        launcher_path = wrapper_output_dir / launcher_name
        launcher_path.write_text("\n".join(launcher_lines) + "\n", encoding="utf-8", newline="\n")
        rendered["launchers"][target_name] = {
            "script": str(launcher_path),
            "mode": launch_mode,
            "controller_window": controller_window,
            "controller_log": controller_log,
            "remote_script": f"{target.remote_tmp_dir}/{launcher_name}",
        }
    else:
        raise ValueError(f"Unsupported launch_mode: {launch_mode}")
    return rendered


def parse_metrics_blob(blob: str) -> dict[str, float]:
    parsed = ast.literal_eval(blob)
    return {key: float(value) for key, value in parsed.items()}


def parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    evaluations = []
    for match in EVAL_RE.finditer(text):
        evaluations.append(
            {
                "step": int(match.group("step")),
                "train_steps": int(match.group("train_steps")),
                "val": parse_metrics_blob(match.group("val")),
                "test_subset": parse_metrics_blob(match.group("test")),
                "selection_source": match.group("source"),
                "best_selection_metric": float(match.group("best")),
            }
        )

    best_subset_match = None
    best_test_match = None
    for match in BEST_SUBSET_RE.finditer(text):
        best_subset_match = match
    for match in BEST_TEST_RE.finditer(text):
        best_test_match = match

    result: dict[str, Any] = {
        "evaluations": evaluations,
        "best_test_subset_metrics": parse_metrics_blob(best_subset_match.group("metrics"))
        if best_subset_match
        else None,
        "best_test_metrics": parse_metrics_blob(best_test_match.group("metrics"))
        if best_test_match
        else None,
    }
    return result


def compare_to_baseline(task_cfg: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    primary = task_cfg["primary_metric"]
    current = float(metrics[primary])
    baseline = float(task_cfg["screening_baseline_metrics"][primary])
    delta = float(task_cfg["screening_metric_delta"])
    noise_floor_delta = float(task_cfg.get("noise_floor_primary_metric_delta", 0.0))
    effective_delta = max(delta, noise_floor_delta)
    higher_is_better = bool(task_cfg["higher_is_better"])
    signed_gain = current - baseline if higher_is_better else baseline - current

    if signed_gain > effective_delta:
        verdict = "better"
    elif signed_gain < -effective_delta:
        verdict = "worse"
    else:
        verdict = "neutral"

    return {
        "primary_metric": primary,
        "current": current,
        "baseline": baseline,
        "signed_gain": signed_gain,
        "delta_threshold": delta,
        "noise_floor_delta": noise_floor_delta,
        "effective_delta_threshold": effective_delta,
        "verdict": verdict,
    }


def compare_to_reference_metrics(task_cfg: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    primary = task_cfg["primary_metric"]
    current = float(metrics[primary])
    baseline = float(task_cfg["full_test_reference_metrics"][primary])
    higher_is_better = bool(task_cfg["higher_is_better"])
    signed_gain = current - baseline if higher_is_better else baseline - current

    if signed_gain > 0:
        verdict = "better"
    elif signed_gain < 0:
        verdict = "worse"
    else:
        verdict = "neutral"

    return {
        "primary_metric": primary,
        "current": current,
        "baseline": baseline,
        "signed_gain": signed_gain,
        "verdict": verdict,
    }


def assess_task_advancement(
    task_cfg: dict[str, Any],
    screening_comparison: dict[str, Any],
    reference_comparison: dict[str, Any],
) -> dict[str, Any]:
    if reference_comparison["verdict"] == "better":
        return {
            "recommendation": "promote",
            "reason": "full-test reference is already beaten on the task primary metric.",
        }

    policy = task_cfg.get("judging_policy", {})
    mode = str(policy.get("retune_plausible_mode", "none"))
    allowed_screening = set(str(item) for item in policy.get("allowed_screening_verdicts", []))
    max_reference_gap = float(policy.get("max_reference_gap", 0.0))
    reason_template = str(policy.get("reason_template", "")).strip()

    def retune(reason: str) -> dict[str, Any]:
        return {
            "recommendation": "retune_plausible",
            "reason": reason,
        }

    if mode == "screening_plus_reference_gap":
        if (
            screening_comparison["verdict"] in allowed_screening
            and reference_comparison["signed_gain"] >= -max_reference_gap
        ):
            return retune(reason_template or "screening signal is positive and the full-test gap remains small.")
    elif mode == "screening_primary":
        if screening_comparison["verdict"] in allowed_screening:
            return retune(reason_template or "task-specific screening behavior remains acceptable.")
    elif mode == "reference_gap":
        if reference_comparison["signed_gain"] >= -max_reference_gap:
            return retune(reason_template or "full-test reference gap remains within the accepted retune band.")

    return {
        "recommendation": "drop",
        "reason": "current task evidence does not justify an Optuna-plus-full-test continuation.",
    }


def judge_candidate(candidate_path: Path, log_dir: Path, write_report: bool) -> dict[str, Any]:
    baselines = load_json(BASELINE_REGISTRY_PATH)["tasks"]
    config = load_json(PIPELINE_CONFIG_PATH)
    candidate = load_json(candidate_path)
    ensure_candidate_valid(candidate, config)

    report: dict[str, Any] = {
        "candidate": str(candidate_path),
        "tasks": {},
    }
    task_verdicts: list[str] = []
    task_recommendations: list[str] = []

    for task_name in config["task_order"]:
        run_id = candidate["task_run_ids"][task_name]
        run_num = exp_num(run_id)
        log_path = log_dir / f"stage3-exp{run_num:03d}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log for {task_name}: {log_path}")

        parsed = parse_log(log_path)
        metrics = parsed["best_test_metrics"] or parsed["best_test_subset_metrics"]
        if not metrics:
            if not parsed["evaluations"]:
                raise ValueError(f"No evaluation lines found in {log_path}")
            metrics = parsed["evaluations"][-1]["test_subset"]

        screening_comparison = compare_to_baseline(baselines[task_name], metrics)
        reference_comparison = compare_to_reference_metrics(baselines[task_name], metrics)
        advancement = assess_task_advancement(
            baselines[task_name],
            screening_comparison,
            reference_comparison,
        )
        task_verdicts.append(reference_comparison["verdict"])
        task_recommendations.append(str(advancement["recommendation"]))
        report["tasks"][task_name] = {
            "run_id": run_id,
            "log_path": str(log_path),
            "metrics": metrics,
            "comparison": screening_comparison,
            "screening_comparison": screening_comparison,
            "reference_comparison": reference_comparison,
            "advancement_assessment": advancement,
        }

    if "worse" in task_verdicts:
        global_verdict = "failed"
    elif "better" in task_verdicts:
        global_verdict = "promotable"
    else:
        global_verdict = "neutral"
    report["global_verdict"] = global_verdict

    if "promote" in task_recommendations and "drop" not in task_recommendations:
        candidate_status = "promotable"
    elif "retune_plausible" in task_recommendations:
        candidate_status = "retune_plausible"
    elif global_verdict == "failed":
        candidate_status = "failed"
    else:
        candidate_status = "stopped"
    report["candidate_status"] = candidate_status

    if write_report:
        dump_json(report_path_for_candidate(candidate_path), report)

    return report


def write_task_launches(candidate_path: Path, launches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate = load_json(candidate_path)
    candidate["task_launches"] = launches
    candidate["status"] = "running"
    dump_json(candidate_path, candidate)
    return candidate


def sync_runtime_files_to_targets(rendered: dict[str, Any], config: dict[str, Any]) -> None:
    targets = build_ssh_targets(config)
    sync_paths = [ROOT / relative_path for relative_path in config.get("launch_sync_paths", ["main.py", "model.py", "utils.py"])]
    gnn_repr_package = ROOT / "gnn_repr"
    if gnn_repr_package.exists():
        sync_paths.append(gnn_repr_package)
    gnn_repr_artifacts = ROOT / "artifacts" / "gnn_repr"
    if gnn_repr_artifacts.exists():
        sync_paths.append(gnn_repr_artifacts)

    files_by_target: dict[str, list[Path]] = {}
    for payload in rendered["wrappers"].values():
        files_by_target.setdefault(payload["target"], [])
        files_by_target[payload["target"]].append(Path(payload["script"]))

    for target_name, wrapper_paths in files_by_target.items():
        target = get_target_by_name(targets, target_name)
        mkdir_proc = ssh_command(
            target,
            f"mkdir -p {target.remote_repo_root} {target.remote_tmp_dir}",
        )
        if mkdir_proc.returncode != 0:
            raise RuntimeError(
                f"Failed to prepare remote directories on {target.name}: {mkdir_proc.stderr.strip()}"
            )

        sync_proc = scp_to_remote_path(target, sync_paths, target.remote_repo_root)
        if sync_proc.returncode != 0:
            raise RuntimeError(f"Code sync failed for {target.name}: {sync_proc.stderr.strip()}")

        launcher_paths = [
            Path(payload["script"])
            for owner, payload in rendered["launchers"].items()
            if owner == target_name
        ]
        upload_paths = wrapper_paths + launcher_paths
        upload_proc = scp_to_remote(target, upload_paths, target.remote_tmp_dir)
        if upload_proc.returncode != 0:
            raise RuntimeError(f"Wrapper sync failed for {target.name}: {upload_proc.stderr.strip()}")


def launch_rendered_candidate(rendered: dict[str, Any], config: dict[str, Any]) -> None:
    targets = build_ssh_targets(config)
    launch_mode = rendered.get("launch_mode", "parallel")
    wrappers_by_target: dict[str, list[dict[str, Any]]] = {}
    for payload in rendered["wrappers"].values():
        wrappers_by_target.setdefault(payload["target"], []).append(payload)

    if launch_mode in {"serial_batch_equivalent", "packed_batch_equivalent"}:
        if len(rendered["launchers"]) != 1:
            raise RuntimeError(f"{launch_mode} launch expects exactly one launcher target.")
        target_name, launcher_payload = next(iter(rendered["launchers"].items()))
        target = get_target_by_name(targets, target_name)
        controller_window = launcher_payload["controller_window"]
        controller_log = launcher_payload["controller_log"]
        proc = ssh_command(
            target,
            " && ".join(
                [
                    f"tmux has-session -t {target.tmux_session} >/dev/null 2>&1 || tmux new-session -d -s {target.tmux_session}",
                    f"tmux kill-window -t {target.tmux_session}:{controller_window} >/dev/null 2>&1 || true",
                    f": > {controller_log}",
                    (
                        f"tmux new-window -d -t {target.tmux_session}: -n {controller_window} "
                        f"\"sh -lc 'bash {launcher_payload['remote_script']} > {controller_log} 2>&1'\""
                    ),
                    f"tmux list-windows -t {target.tmux_session}",
                ]
            ),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Serial launch failed on {target.name}: {proc.stderr.strip()}")
        return

    if launch_mode != "parallel":
        raise RuntimeError(f"Unsupported launch_mode: {launch_mode}")

    for target_name, payloads in wrappers_by_target.items():
        target = get_target_by_name(targets, target_name)
        commands = [
            f"tmux has-session -t {target.tmux_session} >/dev/null 2>&1 || tmux new-session -d -s {target.tmux_session}"
        ]
        for payload in payloads:
            commands.extend(
                [
                    f"tmux kill-window -t {target.tmux_session}:{payload['window_name']} >/dev/null 2>&1 || true",
                    f": > {payload['remote_log']}",
                    (
                        f"tmux new-window -d -t {target.tmux_session}: -n {payload['window_name']} "
                        f"\"sh -lc 'bash {payload['remote_script']} > {payload['remote_log']} 2>&1'\""
                    ),
                ]
            )
        commands.append(f"tmux list-windows -t {target.tmux_session}")
        proc = ssh_command(target, " && ".join(commands))
        if proc.returncode != 0:
            raise RuntimeError(f"Launch failed on {target.name}: {proc.stderr.strip()}")


def print_rendered(rendered: dict[str, Any]) -> None:
    print(f"Candidate: {rendered['candidate']}")
    print(f"Launch mode: {rendered.get('launch_mode', 'parallel')}")
    for task_name, payload in rendered["wrappers"].items():
        world_size = payload.get("launch_world_size", len(normalize_gpu_spec(payload["gpu"])))
        extra = ""
        if payload.get("per_rank_batch_size") is not None:
            extra = f" per_rank_batch={payload['per_rank_batch_size']} world_size={world_size}"
        if payload.get("launch_wave") is not None:
            extra += f" wave={payload['launch_wave']}"
        print(
            f"{task_name}: {payload['run_id']} target={payload['target']} gpu={format_gpu_spec(payload['gpu'])} "
            f"wrapper={payload['script']}{extra}"
        )
    for target_name, launcher_payload in rendered["launchers"].items():
        suffix = ""
        if launcher_payload.get("mode") in {"serial_batch_equivalent", "packed_batch_equivalent"}:
            suffix = (
                f" controller_window={launcher_payload['controller_window']}"
                f" controller_log={launcher_payload['controller_log']}"
            )
        print(f"Launcher[{target_name}]: {launcher_payload['script']}{suffix}")


def print_launch_plan(launches: dict[str, dict[str, Any]]) -> None:
    print("Resolved launch plan:")
    for task_name, payload in launches.items():
        print(
            f"  {task_name}: target={payload['target']} gpu={format_gpu_spec(payload['gpu'])} "
            f"window={payload['window_name']}"
        )


def print_report(report: dict[str, Any]) -> None:
    print(f"Global verdict: {report['global_verdict']}")
    print(f"Candidate status: {report.get('candidate_status', finalized_candidate_status(report['global_verdict']))}")
    for task_name, payload in report["tasks"].items():
        screening = payload["screening_comparison"]
        reference = payload["reference_comparison"]
        advancement = payload.get("advancement_assessment", {})
        print(
            f"{task_name}: reference={reference['verdict']} "
            f"{reference['primary_metric']} current={reference['current']:.6f} "
            f"full_test_ref={reference['baseline']:.6f} gain={reference['signed_gain']:.6f} | "
            f"screening={screening['verdict']} screening_baseline={screening['baseline']:.6f} | "
            f"next={advancement.get('recommendation', 'unknown')}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 research helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_candidate = subparsers.add_parser("new-candidate", help="Create a candidate template.")
    new_candidate.add_argument("--slug", required=True)
    new_candidate.add_argument("--title", required=True)
    new_candidate.add_argument("--family", required=True)
    new_candidate.add_argument(
        "--source-type",
        required=True,
        choices=["paper", "ablation", "paper+ablation"],
    )
    new_candidate.add_argument("--start-exp", required=True, type=int)
    new_candidate.add_argument("--force", action="store_true")

    render = subparsers.add_parser("render", help="Render wrappers and launcher from a candidate spec.")
    render.add_argument("candidate", type=Path)
    render.add_argument("--allow-no-papers", action="store_true")

    launch = subparsers.add_parser(
        "launch",
        help="Auto-assign idle GPUs across configured targets, sync code, and launch one candidate bundle.",
    )
    launch.add_argument("candidate", type=Path)
    launch.add_argument("--allow-no-papers", action="store_true")
    launch.add_argument("--dry-run", action="store_true")

    parse = subparsers.add_parser("parse-log", help="Parse one stage3 log file.")
    parse.add_argument("log", type=Path)

    judge = subparsers.add_parser("judge", help="Judge a candidate bundle against strict baselines.")
    judge.add_argument("candidate", type=Path)
    judge.add_argument("--log-dir", type=Path, required=True)
    judge.add_argument("--write-report", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_json(PIPELINE_CONFIG_PATH)

    if args.command == "new-candidate":
        slug = sanitize_slug(args.slug)
        candidate = make_candidate_template(args, config)
        path = ROOT / config["candidate_dir"] / candidate_filename(args.start_exp, slug)
        if path.exists() and not args.force:
            raise FileExistsError(f"Candidate file already exists: {path}")
        dump_json(path, candidate)
        print(path)
        return

    if args.command == "render":
        rendered = render_candidate(args.candidate, allow_no_papers=args.allow_no_papers)
        print_rendered(rendered)
        return

    if args.command == "launch":
        launches = assign_candidate_launches(args.candidate, config)
        print_launch_plan(launches)
        rendered = render_candidate(
            args.candidate,
            allow_no_papers=args.allow_no_papers,
            task_launches=launches,
        )
        print_rendered(rendered)
        if args.dry_run:
            return
        sync_runtime_files_to_targets(rendered, config)
        launch_rendered_candidate(rendered, config)
        write_task_launches(args.candidate, launches)
        print("Launch completed.")
        return

    if args.command == "parse-log":
        parsed = parse_log(args.log)
        print(json.dumps(parsed, indent=2))
        return

    if args.command == "judge":
        report = judge_candidate(args.candidate, args.log_dir, write_report=args.write_report)
        if args.write_report:
            update_candidate_status_from_report(args.candidate, report)
        print_report(report)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
