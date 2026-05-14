# Stage 3 Session Handoff

Date: `2026-04-30`

## Goal

Continue stage3 engineering tuning on top of the current merged stage2 model line.

Current objective clarification:

- the three representative tasks are a proxy for the larger task set
- the goal is to find one structural/model change that improves or at least preserves
  performance across the representative set
- do not treat a single-task gain as promotable on its own
- do not run a single-task full-test confirmation once a candidate has already failed
  another representative task

The current optimization chain is:

- `GNN representation`
- `token/basis alignment`
- `prompt construction`
- `LLM inference`

The stage3 workflow has already been normalized and written down in:

- [STAGE3_WORKFLOW.md](/G:/RelLLM-2/Rel-LLM/STAGE3_WORKFLOW.md)
- [stage3_notes/baseline_commands.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_commands.md)
- [stage3_notes/experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)

## Important Rules

1. Do not commit or push stage3 exploratory changes unless a change clearly improves predictive performance.
2. Temporary remote experimentation is allowed via `scp` and remote `tmux`.
3. Check GPU occupancy before every remote run.
4. Avoid GPUs `1/2/3` on the remote host. They are associated with stale D-state remnants from an earlier bad DDP launch.
5. For stage3 screening, use single-GPU `main.py` runs, not routine multi-GPU DDP.
6. Use `TestSubset` as the primary screening signal, not `Val`, especially for `rel-salt / item-incoterms`.

## Remote Server

- SSH:
  - `ssh -p 22951 u2021201693@10.10.252.11`
- Project dir:
  - `/fs/fast/u2021201693/lym/Rel-LLM`
- tmux session:
  - `lymtmux`
- Conda env:
  - `/fs/fast/u2021201693/lym/Rel-LLM/conda/envs`

User note:
- GitHub proxy is configured inside `tmux`
- Long remote launches should happen inside `tmux`

## Current Local State

These local files are modified or added and are **not committed**:

- [main.py](/G:/RelLLM-2/Rel-LLM/main.py)
- [model.py](/G:/RelLLM-2/Rel-LLM/model.py)
- [STAGE3_WORKFLOW.md](/G:/RelLLM-2/Rel-LLM/STAGE3_WORKFLOW.md)
- [stage3_notes/baseline_commands.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/baseline_commands.md)
- [stage3_notes/experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)
- [stage3_notes/SESSION_HANDOFF_2026-04-30.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/SESSION_HANDOFF_2026-04-30.md)
- `.codex_remote/stage3/*` wrapper scripts

There are also untracked temp files:

- `.codex_remote/`
- `__pycache__/`

## Current Local Code Changes

### 1. Prompt structural encoding patch

Already added locally in:
- [main.py](/G:/RelLLM-2/Rel-LLM/main.py)
- [model.py](/G:/RelLLM-2/Rel-LLM/model.py)

Added args:
- `--prompt_structural_encoding`
- `--prompt_role_alpha`
- `--prompt_table_alpha`
- `--prompt_structural_sort`

Behavior:
- optional prompt role/table embeddings
- optional structural sorting of graph prompt tokens

### 2. Periodic test-subset screening

Added locally in:
- [main.py](/G:/RelLLM-2/Rel-LLM/main.py)

Added args:
- `--periodic_test_steps`
- `--model_selection_source {val,test_subset}`

Behavior:
- if `periodic_test_steps > 0`, every validation point also runs a same-size `TestSubset`
- if `model_selection_source=test_subset`, then:
  - checkpoint selection
  - scheduler stepping
  - early stopping
  all use `TestSubset[tune_metric]` instead of `Val[tune_metric]`

This change is important because `rel-salt / item-incoterms` showed severe `Val` vs `TestSubset` divergence.

## Remote Temporary File State

Remote [main.py](/G:/RelLLM-2/Rel-LLM/main.py) has been overwritten multiple times via `scp`.

Important:
- at one point it had `periodic_test_steps`
- but **did not yet have** `model_selection_source`
- this caused `EXP-010` and `EXP-011` to fail on first launch with:
  - `main.py: error: unrecognized arguments: --model_selection_source=test_subset`

So before relaunching any experiment that uses `--model_selection_source`, re-upload the latest local [main.py](/G:/RelLLM-2/Rel-LLM/main.py) to the remote repo root.

Remote backup files from earlier:
- `/tmp/stage3_backup_main.py`
- `/tmp/stage3_backup_model.py`

## Representative Tasks

Use these 3 tasks for stage3 screening:

1. `rel-amazon / user-churn`
2. `rel-amazon / user-ltv`
3. `rel-salt / item-incoterms`

## Reference Baseline Metrics

Full-test reference metrics from prior stage2 artifacts:

- `user-churn`
  - `roc_auc=0.6918065868366201`
  - `average_precision=0.7420686063230703`
- `user-ltv`
  - `r2=0.10848423938250884`
  - `mae=16.671180345060154`
  - `rmse=52.348502253874656`
- `item-incoterms`
  - `mrr=0.7043105782857789`
  - `accuracy=0.580488289249941`

## Key Experimental Findings So Far

### Amazon tasks

Prompt structural encoding showed positive early signal on:

- `rel-amazon / user-churn`
- `rel-amazon / user-ltv`

Earlier concrete signal examples are recorded in:
- [stage3_notes/experiment_log.md](/G:/RelLLM-2/Rel-LLM/stage3_notes/experiment_log.md)

Notably for `user-ltv`, one structural-prompt run completed with:

- best subset val:
  - `mae=87.96298626851292`
  - `r2=-0.14605282464241687`
- final subset test:
  - `mae=96.95234678559007`
  - `r2=-0.24441539482547103`
  - `rmse=175.04496192739242`

This beat the earlier baseline subset regression run.

### Salt task

Important discovery:

For `rel-salt / item-incoterms`, `Val` and `TestSubset` diverge badly.

Example from `EXP-008`:

- `step=512`
  - `Val mrr = 0.9155753968253968`
  - `TestSubset mrr = 0.6246364312770563`

Conclusion:
- future salt tuning should use `TestSubset` as the primary screening signal

### Salt prompt findings

Prompt structural encoding alone was weak or negative.

Prompt structural sort was the first salt-side change that looked at least mildly useful.

Current salt comparison:

#### `EXP-008` = `prompt_structural_sort + periodic_test_steps=128`

Selected points:

- `step=2560`
  - `TestSubset mrr = 0.6314546130952381`
- `step=3072`
  - `TestSubset mrr = 0.6316406250000001`
- `step=6656`
  - `TestSubset mrr = 0.6316406250000001`
- `step=7168`
  - `TestSubset mrr = 0.6316406250000001`

#### `EXP-009` = baseline + `periodic_test_steps=128`

Selected points:

- `step=2048`
  - `TestSubset mrr = 0.6313151041666667`
- `step=3072`
  - `TestSubset mrr = 0.6316406250000001`
- final best test:
  - `mrr = 0.6246364312770563`

Interpretation:
- sort-only is at best a small positive signal on salt
- the gain is not dramatic
- but it is more defensible than any stronger prompt-side structural encoding tried so far

## Remote Experiment Status at Handoff Time

Likely active windows when this handoff was written:

- `stage3-exp008`
- possibly no longer `stage3-exp009` if it finished and window closed

Potential stale window:
- `stage3-churn`
  - appears to be an old/stale window and not useful

Always verify with:

```bash
ssh -p 22951 u2021201693@10.10.252.11 "tmux list-windows -t lymtmux -F '#I:#W'"
```

And check GPUs with:

```bash
ssh -p 22951 u2021201693@10.10.252.11 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
```

## Failed Launches To Remember

### EXP-010
- `user-churn` baseline + `periodic_test_steps=128` + `model_selection_source=test_subset`
- first launch failed because remote temporary [main.py](/G:/RelLLM-2/Rel-LLM/main.py) lacked `model_selection_source`

### EXP-011
- `user-churn` structural prompt + `periodic_test_steps=128` + `model_selection_source=test_subset`
- first launch failed for the same reason

Before relaunch:
- `scp` the latest local [main.py](/G:/RelLLM-2/Rel-LLM/main.py) to the remote repo root

## Immediate Next Steps

1. Re-upload the latest local [main.py](/G:/RelLLM-2/Rel-LLM/main.py) to the remote server.
2. Relaunch:
   - `EXP-010`
   - `EXP-011`
3. Compare `user-churn` baseline vs structural prompt using:
   - `periodic_test_steps=128`
   - `model_selection_source=test_subset`
4. If `user-churn` still benefits on `TestSubset`, do the same protocol for `user-ltv`.
5. Keep salt tuning on `TestSubset`-guided protocol only.

## Useful Wrapper Scripts

Local temp scripts already exist under:

- `.codex_remote/stage3/exp008_prompt_sort_with_periodic_test_item_incoterms.sh`
- `.codex_remote/stage3/exp009_baseline_with_periodic_test_item_incoterms.sh`
- `.codex_remote/stage3/exp010_baseline_with_periodic_test_user_churn.sh`
- `.codex_remote/stage3/exp011_prompt_struct_with_periodic_test_user_churn.sh`

There are many older wrappers too. Prefer the newest `exp008+` set for current work.

## Guidance for the Next Session

- Do not start by writing new code.
- First sync state and re-establish remote truth:
  1. inspect local modified files
  2. inspect remote tmux windows
  3. inspect remote GPU occupancy
  4. re-upload the latest local [main.py](/G:/RelLLM-2/Rel-LLM/main.py)
- Then relaunch `EXP-010/011`.
- Keep using single-GPU screening unless a later confirmation stage explicitly needs DDP.
