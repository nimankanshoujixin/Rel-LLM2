# Stage 3 Session Handoff

Date: `2026-05-05`

## Goal

Continue Stage 3 tuning on top of the current Stage 2 merged model line, but do it through
the new programmatic research pipeline rather than through ad hoc manual experiment driving.

Current objective:

- the 3 representative tasks are a proxy set for the broader task pool
- a candidate must be non-regressive across:
  - `rel-amazon / user-churn`
  - `rel-amazon / user-ltv`
  - `rel-salt / item-incoterms`
- a single-task gain is not promotable on its own
- paper-backed or at least ablation-backed candidate generation is now required

## Read These First

1. [STAGE3_WORKFLOW.md](/G:/RelLLM-2/Rel-LLM/STAGE3_WORKFLOW.md)
2. [stage3_notes/STAGE3_PROGRAM.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/STAGE3_PROGRAM.md)
3. [stage3_notes/experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)
4. [stage3_notes/baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)
5. [stage3_notes/pipeline_config.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/pipeline_config.json)
6. [stage3_notes/paper_shortlist_2026-05-05.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/paper_shortlist_2026-05-05.md)

## What Changed This Session

The main progress was not a new model win. It was a pipeline upgrade.

New local tooling:

- [stage3_research.py](/G:/RelLLM-2/Rel-LLM/stage3_research.py)
  - creates machine-readable candidate specs
  - renders wrappers and tmux launchers
  - parses logs
  - judges bundles against strict screening baselines

- [stage3_orchestrator.py](/G:/RelLLM-2/Rel-LLM/stage3_orchestrator.py)
  - runs locally
  - checks remote `tmux` windows and GPU state via SSH
  - syncs `/tmp/stage3-exp*.log` back to local cache
  - prints bundle status locally
  - recommends early kill
  - auto-judges completed bundles

New machine-readable state:

- [stage3_notes/candidates/README.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/README.md)
- [stage3_notes/candidates/exp036_basis_loss_only.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp036_basis_loss_only.json)
- [stage3_notes/candidates/exp039_basis_token_only.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp039_basis_token_only.json)
- [stage3_notes/candidates/exp042_paper_guided_basis_scan.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp042_paper_guided_basis_scan.json)
- [stage3_notes/reports/exp036_basis_loss_only.report.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/reports/exp036_basis_loss_only.report.json)
- [stage3_notes/reports/exp039_basis_token_only.report.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/reports/exp039_basis_token_only.report.json)

## Remote Server

- SSH:
  - `ssh -p 22951 u2021201693@10.10.252.11`
- Project dir:
  - `/fs/fast/u2021201693/lym/Rel-LLM`
- tmux session:
  - `lymtmux`
- Remote temp/log dir:
  - `/tmp`

Avoid GPUs `1/2/3`.

As of this handoff, representative-task experiment windows are no longer running; the last
checked `lymtmux` state only had:

- `bash`
- old `stage3-churn`

Always verify remote truth again before acting.

## Current Representative Baselines

Machine-readable source:

- [stage3_notes/baseline_registry.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_registry.json)

Strict screening baselines:

- `user-churn` (`EXP-010`)
  - `roc_auc=0.6352813852813852`
  - `average_precision=0.7769216138610865`

- `user-ltv` (`EXP-014`)
  - `mae=100.98803919482977`
  - `r2=-0.3492479376404152`
  - `rmse=182.26899940576845`

- `item-incoterms` (`EXP-012`)
  - `mrr=0.6311197916666667`
  - `accuracy=0.5078125`

## Latest Experimental Truth

Recent candidate families already failed:

1. projector LayerNorm
   - `EXP-030/031/032`
   - failed globally

2. no-basis ablation
   - `EXP-033/034/035`
   - failed globally

3. basis-loss-only
   - `EXP-036/037/038`
   - failed globally

4. token-only basis residual
   - `EXP-039/040/041`
   - judged failed globally
   - details are in:
     - [exp039_basis_token_only.report.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/reports/exp039_basis_token_only.report.json)

Important detail from `EXP-039/040/041`:

- `user-ltv` improved
- `item-incoterms` was neutral
- `user-churn` regressed enough to kill the bundle

That pattern reinforces the core rule:

- do not promote partial wins

## Current Best Next Direction

Do **not** continue scanning tiny prompt or basis coefficients blindly.

The next candidate should be:

- paper-backed
- represented as a candidate bundle JSON first
- rendered by `stage3_research.py`
- monitored by `stage3_orchestrator.py`

Best current draft entry point:

- [stage3_notes/candidates/exp042_paper_guided_basis_scan.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp042_paper_guided_basis_scan.json)

This file is only a scaffold right now and still needs:

- real literature queries
- relevant papers
- explicit mechanism hypothesis
- concrete overrides

## Recommended Next Session Actions

1. Inspect local state:
   - `git status --short`
2. Inspect remote truth:
   - `python stage3_orchestrator.py list-candidates`
   - `python stage3_orchestrator.py status stage3_notes/candidates/exp039_basis_token_only.json --sync-logs --recommend-kill`
3. Treat `EXP-039/040/041` as settled failed history.
4. Fill [exp042_paper_guided_basis_scan.json](/G:/RelLLM-2/Rel-LLM/stage3_notes/candidates/exp042_paper_guided_basis_scan.json) based on the paper shortlist.
5. Render its wrappers:
   - `python stage3_research.py render stage3_notes/candidates/exp042_paper_guided_basis_scan.json`
6. Launch the resulting bundle and monitor it locally with:
   - `python stage3_orchestrator.py monitor stage3_notes/candidates/exp042_paper_guided_basis_scan.json --interval-sec 60 --write-report`

## Continuous Progression Rule

Do not treat completion of the currently monitored bundle as the end of the Stage 3 task.

After a bundle finishes, the next thread / heartbeat must:

1. sync logs and run the judge/report path
2. update the candidate JSON and `experiment_log.md`
3. record the search-space consequence
4. choose the next paper- or ablation-backed Stage 3 action
5. launch the next justified bundle if it clears static / sanity gates, or retarget/create a
   heartbeat for the active non-launch next step

Delete a heartbeat only when Stage 3 is explicitly paused/complete, or after a replacement
heartbeat has been created for the next active step.

## Notes

- Do not commit or push exploratory Stage 3 changes unless there is a real global gain.
- `main.py` and `model.py` are still locally modified and uncommitted.
- `.codex_remote/`, `stage3_notes/`, `stage3_research.py`, and `stage3_orchestrator.py`
  are also uncommitted local state.
