from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from stage3_research import (
    PIPELINE_CONFIG_PATH,
    SshTarget,
    build_ssh_targets,
    candidate_launch_mode,
    compare_to_baseline,
    get_target_by_name,
    BASELINE_REGISTRY_PATH,
    parse_log,
    query_gpus,
    query_tmux_windows,
    report_path_for_candidate,
    scp_from_remote,
    ssh_command,
    update_candidate_status_from_report,
    exp_num,
    judge_candidate,
    load_json,
)


ROOT = Path(__file__).resolve().parent


def load_pipeline_config() -> dict[str, Any]:
    return load_json(PIPELINE_CONFIG_PATH)


def list_candidate_specs(config: dict[str, Any]) -> list[Path]:
    candidate_dir = ROOT / config["candidate_dir"]
    return sorted(candidate_dir.glob("*.json"))


def load_candidate(path: Path) -> dict[str, Any]:
    return load_json(path)


def run_id_to_window(run_id: str, prefix: str) -> str:
    return f"{prefix}{exp_num(run_id):03d}"


def candidate_launches(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = build_ssh_targets(config)
    default_target = targets[0]
    launches = {}
    for task_name, run_id in candidate["task_run_ids"].items():
        payload = candidate.get("task_launches", {}).get(task_name, {})
        target_name = payload.get("target", default_target.name)
        target = get_target_by_name(targets, target_name)
        launches[task_name] = {
            "target": target.name,
            "window_name": payload.get("window_name", run_id_to_window(run_id, config["window_prefix"])),
            "remote_log": payload.get("remote_log", f"{target.remote_tmp_dir}/stage3-exp{exp_num(run_id):03d}.log"),
            "gpu": payload.get("gpu"),
        }
    return launches


def sync_candidate_logs(
    candidate: dict[str, Any],
    config: dict[str, Any],
    log_cache_dir: Path | None = None,
) -> tuple[Path, list[Path]]:
    launches = candidate_launches(candidate, config)
    targets = build_ssh_targets(config)
    local_dir = log_cache_dir or (ROOT / config["log_cache_dir"])
    for payload in launches.values():
        target = get_target_by_name(targets, payload["target"])
        proc = scp_from_remote(target, [payload["remote_log"]], local_dir)
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            if "No such file or directory" in stderr:
                continue
            raise RuntimeError(f"scp failed for {target.name}: {stderr}")
    downloaded = []
    for run_id in candidate["task_run_ids"].values():
        downloaded.append(local_dir / f"stage3-exp{exp_num(run_id):03d}.log")
    return local_dir, downloaded


def detect_completion(
    candidate: dict[str, Any],
    windows_by_target: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> bool:
    launches = candidate_launches(candidate, config)
    for payload in launches.values():
        target_windows = windows_by_target.get(payload["target"], {})
        if payload["window_name"] in target_windows:
            return False
    return True


def read_local_log_tail(path: Path, line_limit: int = 5) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-line_limit:]


def latest_eval_step(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(
        __import__("re").finditer(r"\[Eval\] Train step: (\d+)/", text)
    )
    if not matches:
        return None
    return int(matches[-1].group(1))


def candidate_log_paths(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Path]:
    log_dir = ROOT / config["log_cache_dir"]
    return {
        task_name: log_dir / f"stage3-exp{exp_num(run_id):03d}.log"
        for task_name, run_id in candidate["task_run_ids"].items()
    }


def can_finalize_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> bool:
    return all(path.exists() for path in candidate_log_paths(candidate, config).values())


def can_judge_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> bool:
    for log_path in candidate_log_paths(candidate, config).values():
        if not log_path.exists():
            return False
        parsed = parse_log(log_path)
        if parsed["best_test_metrics"] or parsed["best_test_subset_metrics"] or parsed["evaluations"]:
            continue
        return False
    return True


def summarize_candidate_state(
    candidate_path: Path,
    config: dict[str, Any],
    sync_logs: bool,
) -> dict[str, Any]:
    candidate = load_candidate(candidate_path)
    targets = build_ssh_targets(config)
    windows_by_target = {
        target.name: query_tmux_windows(target)
        for target in targets
    }
    gpus_by_target = {
        target.name: query_gpus(target)
        for target in targets
    }
    local_dir = ROOT / config["log_cache_dir"]
    if sync_logs:
        sync_candidate_logs(candidate, config, log_cache_dir=local_dir)

    launches = candidate_launches(candidate, config)
    task_states = {}
    for task_name, run_id in candidate["task_run_ids"].items():
        launch = launches[task_name]
        window_name = launch["window_name"]
        log_path = local_dir / f"stage3-exp{exp_num(run_id):03d}.log"
        task_states[task_name] = {
            "run_id": run_id,
            "target": launch["target"],
            "gpu": launch.get("gpu"),
            "window_name": window_name,
            "window_present": window_name in windows_by_target.get(launch["target"], {}),
            "latest_eval_step": latest_eval_step(log_path),
            "log_path": str(log_path),
            "log_tail": read_local_log_tail(log_path),
        }

    completed = detect_completion(candidate, windows_by_target, config)
    return {
        "candidate_path": str(candidate_path),
        "title": candidate.get("title", candidate_path.stem),
        "status": candidate.get("status", "unknown"),
        "completed": completed,
        "task_states": task_states,
        "gpus_by_target": gpus_by_target,
    }


def recommend_early_kill(candidate_path: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    candidate = load_candidate(candidate_path)
    log_dir = ROOT / config["log_cache_dir"]
    task_order = config["task_order"]
    launch_mode = candidate_launch_mode(candidate)
    if launch_mode == "serial_batch_equivalent":
        state = summarize_candidate_state(candidate_path, config, sync_logs=False)
        baselines = load_json(BASELINE_REGISTRY_PATH)["tasks"]
        for task_name in task_order:
            task_state = state["task_states"][task_name]
            if task_state["window_present"]:
                return {
                    "recommendation": "keep_watching",
                    "reason": f"{task_name} is still the active serial task.",
                    "report": None,
                }
            log_path = Path(task_state["log_path"])
            if not log_path.exists():
                return {
                    "recommendation": "keep_watching",
                    "reason": f"{task_name} has not completed yet under serial scheduling.",
                    "report": None,
                }
            parsed = parse_log(log_path)
            metrics = parsed["best_test_metrics"] or parsed["best_test_subset_metrics"]
            if not metrics:
                if not parsed["evaluations"]:
                    return None
                metrics = parsed["evaluations"][-1]["test_subset"]
            comparison = compare_to_baseline(baselines[task_name], metrics)
            if comparison["verdict"] == "worse":
                return {
                    "recommendation": "kill_bundle",
                    "reason": (
                        f"{task_name} is already worse under the effective screening delta "
                        f"({comparison['effective_delta_threshold']:.6f}); later serial tasks can be skipped."
                    ),
                    "report": {
                        "task_name": task_name,
                        "metrics": metrics,
                        "comparison": comparison,
                    },
                }
    if launch_mode == "packed_batch_equivalent":
        state = summarize_candidate_state(candidate_path, config, sync_logs=False)
        active_tasks = [
            task_name
            for task_name in task_order
            if state["task_states"][task_name]["window_present"]
        ]
        if active_tasks:
            return {
                "recommendation": "keep_watching",
                "reason": f"packed wave active tasks: {', '.join(active_tasks)}.",
                "report": None,
            }
        return {
            "recommendation": "keep_watching",
            "reason": "packed batch-equivalent bundle has no active task windows at this poll.",
            "report": None,
        }
    try:
        report = judge_candidate(candidate_path, log_dir, write_report=False)
    except Exception:
        return None

    first_task = task_order[0]
    first_verdict = report["tasks"][first_task]["comparison"]["verdict"]
    if first_verdict == "worse":
        return {
            "recommendation": "kill_bundle",
            "reason": f"{first_task} is already worse on the primary screening metric.",
            "report": report,
        }
    return {
        "recommendation": "keep_watching",
        "reason": f"{first_task} has not yet triggered an early-kill condition.",
        "report": report,
    }


def kill_candidate_windows(candidate_path: Path, config: dict[str, Any]) -> None:
    candidate = load_candidate(candidate_path)
    launches = candidate_launches(candidate, config)
    targets = build_ssh_targets(config)
    commands_by_target: dict[str, list[str]] = {}
    for payload in launches.values():
        target = get_target_by_name(targets, payload["target"])
        commands_by_target.setdefault(target.name, []).append(
            f"tmux kill-window -t {target.tmux_session}:{payload['window_name']} >/dev/null 2>&1 || true"
        )
    for target_name, commands in commands_by_target.items():
        target = get_target_by_name(targets, target_name)
        ssh_command(target, "; ".join(commands))
    if candidate_launch_mode(candidate) in {"serial_batch_equivalent", "packed_batch_equivalent"}:
        first_task = config["task_order"][0]
        first_launch = launches[first_task]
        target = get_target_by_name(targets, first_launch["target"])
        suffix = "packed" if candidate_launch_mode(candidate) == "packed_batch_equivalent" else "serial"
        controller_window = f"stage3-bundle-exp{exp_num(candidate['task_run_ids'][first_task]):03d}-{suffix}"
        ssh_command(
            target,
            f"tmux kill-window -t {target.tmux_session}:{controller_window} >/dev/null 2>&1 || true",
        )


def finalize_candidate(candidate_path: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    report = judge_candidate(
        candidate_path,
        ROOT / config["log_cache_dir"],
        write_report=True,
    )
    candidate = update_candidate_status_from_report(candidate_path, report)
    return report, candidate


def print_gpu_table(target_name: str, gpus: list[dict[str, Any]]) -> None:
    print(f"GPU status [{target_name}]:")
    for row in gpus:
        idle_reason = row.get("idle_reason")
        idle_suffix = f" idle={idle_reason}" if idle_reason else ""
        print(
            f"  gpu={row['index']} mem={row['memory_used_mb']}MB util={row['utilization_gpu']}%{idle_suffix}"
        )


def print_candidate_state(state: dict[str, Any]) -> None:
    print(f"Candidate: {state['candidate_path']}")
    print(f"Title: {state['title']}")
    print(f"Status: {state['status']}")
    print(f"Completed: {state['completed']}")
    for task_name, payload in state["task_states"].items():
        print(
            f"  {task_name}: {payload['run_id']} "
            f"target={payload['target']} gpu={payload['gpu']} "
            f"window={'up' if payload['window_present'] else 'down'} "
            f"latest_eval={payload['latest_eval_step']}"
        )
        for tail_line in payload["log_tail"]:
            print(f"    tail: {tail_line}")
    for target_name, gpus in state["gpus_by_target"].items():
        print_gpu_table(target_name, gpus)


def cmd_status(args: argparse.Namespace) -> int:
    config = load_pipeline_config()
    state = summarize_candidate_state(args.candidate, config, sync_logs=args.sync_logs)
    print_candidate_state(state)
    if state["completed"] and args.sync_logs:
        candidate = load_candidate(args.candidate)
        if not can_finalize_candidate(candidate, config):
            print("Remote windows are down, but logs are not fully synchronized yet. Skipping finalize for now.")
            return 0
        if not can_judge_candidate(candidate, config):
            print("Remote windows are down, but at least one task has no evaluation lines yet. Skipping finalize for now.")
            return 0
        print("Remote windows are down. Finalizing completed bundle.")
        report, candidate = finalize_candidate(args.candidate, config)
        print(json.dumps(report, indent=2))
        print(f"Candidate status updated to: {candidate.get('status', 'unknown')}")
        print(f"Report written to: {report_path_for_candidate(args.candidate)}")
        return 0
    if args.recommend_kill:
        recommendation = recommend_early_kill(args.candidate, config)
        if recommendation:
            print(f"Recommendation: {recommendation['recommendation']}")
            print(f"Reason: {recommendation['reason']}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    config = load_pipeline_config()
    interval = args.interval_sec
    iteration = 0

    while True:
        iteration += 1
        print(f"\n=== monitor iteration {iteration} ===")
        state = summarize_candidate_state(args.candidate, config, sync_logs=True)
        print_candidate_state(state)

        recommendation = recommend_early_kill(args.candidate, config)
        if recommendation:
            print(f"Recommendation: {recommendation['recommendation']}")
            print(f"Reason: {recommendation['reason']}")
            if args.kill_failed and recommendation["recommendation"] == "kill_bundle":
                print("Action: killing candidate windows on remote.")
                kill_candidate_windows(args.candidate, config)
                state = summarize_candidate_state(args.candidate, config, sync_logs=True)
                print_candidate_state(state)
                if state["completed"]:
                    candidate = load_candidate(args.candidate)
                    if not can_finalize_candidate(candidate, config):
                        print("Remote windows are down after kill, but logs are not fully synchronized yet.")
                        return 0
                    if not can_judge_candidate(candidate, config):
                        print("Remote windows are down after kill, but at least one task has no evaluation lines yet.")
                        return 0
                    print("Remote windows are down after kill. Finalizing completed bundle.")
                    report, candidate = finalize_candidate(args.candidate, config)
                    print(json.dumps(report, indent=2))
                    print(f"Candidate status updated to: {candidate.get('status', 'unknown')}")
                    print(f"Report written to: {report_path_for_candidate(args.candidate)}")
                return 0

        if state["completed"]:
            candidate = load_candidate(args.candidate)
            if not can_finalize_candidate(candidate, config):
                print("Remote windows are down, but logs are not fully synchronized yet. Waiting for next poll.")
                return 0
            if not can_judge_candidate(candidate, config):
                print("Remote windows are down, but at least one task has no evaluation lines yet. Waiting for next poll.")
                return 0
            print("Remote windows are down. Judging completed bundle.")
            report, candidate = finalize_candidate(args.candidate, config)
            print(json.dumps(report, indent=2))
            print(f"Candidate status updated to: {candidate.get('status', 'unknown')}")
            print(f"Report written to: {report_path_for_candidate(args.candidate)}")
            return 0

        if args.iterations and iteration >= args.iterations:
            return 0
        time.sleep(interval)


def cmd_list_candidates(args: argparse.Namespace) -> int:
    config = load_pipeline_config()
    for path in list_candidate_specs(config):
        candidate = load_candidate(path)
        print(f"{path} :: {candidate.get('status', 'unknown')} :: {candidate.get('title', path.stem)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Stage 3 orchestrator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list-candidates", help="List candidate bundle specs.")
    list_cmd.set_defaults(func=cmd_list_candidates)

    status_cmd = subparsers.add_parser("status", help="Query remote state for one candidate.")
    status_cmd.add_argument("candidate", type=Path)
    status_cmd.add_argument("--sync-logs", action="store_true")
    status_cmd.add_argument("--recommend-kill", action="store_true")
    status_cmd.set_defaults(func=cmd_status)

    monitor_cmd = subparsers.add_parser("monitor", help="Poll remote state until completion or stop condition.")
    monitor_cmd.add_argument("candidate", type=Path)
    monitor_cmd.add_argument("--interval-sec", type=int, default=60)
    monitor_cmd.add_argument("--iterations", type=int, default=0)
    monitor_cmd.add_argument("--kill-failed", action="store_true")
    monitor_cmd.add_argument("--write-report", action="store_true")
    monitor_cmd.set_defaults(func=cmd_monitor)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
