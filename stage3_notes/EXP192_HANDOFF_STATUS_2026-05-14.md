# EXP192 Handoff Status

Date: 2026-05-14

Scope:

- clean worktree only: `G:\RelLLM-2\Rel-LLM-clean-p13`
- branch: `codex/stage3-clean-p13`
- remote target: `lab25211`

## Verified local state

- continuing from the clean worktree only
- local Stage 3 context files, candidate JSON, and report were re-read before action
- `stage3_notes/pipeline_config.json` was updated so remote launch sync now includes:
  - `tune_hyperparameters.py`
  - `main.py`
  - `model.py`
  - `text_embedder.py`
  - `utils.py`

## Verified remote state on takeover

- remote repo path remains `/fs/fast/u2021201693/lym/Rel-LLM`
- remote branch/worktree is not the clean local branch; treat remote as a run target only
- `lymtmux` no longer had the older Stage 3 Optuna controller windows alive at takeover
- therefore old tmux state was not trusted and was re-verified from scratch

## Verified prior Optuna outcomes

- `optuna_runs/exp192_user_churn_optuna_20260513t2307`
  - completed
  - contains `best_trial.json`
  - recorded `best_value=0.6952360855480397`
  - selected `batch_size=3` with 2-rank DDP
- `optuna_runs/exp194_item_incoterms_optuna_20260513t2307`
  - completed
  - contains `best_trial.json`
  - recorded `best_value=0.7080100780296092`
  - selected `batch_size=4` with 2-rank DDP
- `optuna_runs/exp192_user_churn_optuna_20260513t2248`
  - incomplete early attempt
  - only `trial_0000.log`
  - no `best_trial.json`

Interpretation:

- the previously running `user-churn` and `item-incoterms` Optuna studies did not survive as live
  tmux-managed jobs, but they had already finished and left artifacts
- on 2026-05-14 the tuning rule was clarified: Optuna phase does **not** need batch-equivalent
  constraints, because `batch_size` is itself part of the hyperparameter search
- therefore the completed `EXP192` / `EXP194` Optuna studies remain valid tuning outputs and should
  be judged on their own search protocol correctness rather than rejected for larger selected batch
  sizes

## Active action taken on 2026-05-14

- created fresh heartbeat automation:
  - name: `Monitor EXP192 Optuna Continuation`
  - id: `monitor-exp192-optuna-continuation`
- launched a new bounded `user-ltv` Optuna run:
  - initial study: `exp193_user_ltv_optuna_20260514t1357`
  - storage: `sqlite:///stage3_optuna_exp193_user_ltv_20260514t1357.db`
  - tmux window: `stage3-exp193-optuna`
  - remote log: `/tmp/stage3-exp193-optuna.log`
  - GPUs: `1,2`
  - `nproc_per_node=2`
  - `max_gpus_per_task=2`
  - `periodic_test_steps=512`
  - `model_selection_source=test_subset`
  - no final full test
  - bounded budget: `12` trials
  - the first relaunch mistakenly constrained `batch-size-choices` to `1,2` under an outdated
    batch-equivalent assumption and must be replaced by a fresh unconstrained Optuna launch

## Immediate launch/debug notes

- first relaunch failed because the remote wrapper had line-ending pollution and argparse saw
  `--debug` as an unrecognized `--debug\\r`
- fixed by normalizing the remote wrapper to LF and relaunching under tmux
- user clarified on 2026-05-14 that Optuna does **not** need batch-equivalent constraints, because
  `batch_size` is part of the hyperparameter search itself
- therefore the mistakenly constrained `exp193_user_ltv_optuna_20260514t1357` relaunch was stopped
  and replaced
- active replacement study:
  - transient retry `exp193_user_ltv_optuna_20260514t141227` exposed two immediate blockers and was
    retired
  - active study is now `exp193_user_ltv_optuna_20260514t141722`
  - same 2-GPU cap and Stage 3 subset protocol
  - no manual batch-equivalent restriction on `batch_size`
- current verified live process chain after the replacement relaunch:
  - wrapper bash process
  - `tune_hyperparameters.py`
  - later trial subprocesses should appear under `torch.distributed.run` and two `main.py` ranks
- blocker resolved during same-session debug:
  - when `channels` was left unconstrained, Optuna sampled `64` and immediately failed because the
    Amazon GNN representation artifact is fixed at `256` channels
  - the valid correction is to keep Optuna unconstrained on batch size but constrain artifact-bound
    structural knobs to the compatible space:
    - `channels-choices 256`
    - `num-layers-choices 2`
    - `aggr-choices mean`
- second blocker resolved during same-session debug:
  - reusing the same study name after changing categorical search choices caused Optuna
    `CategoricalDistribution does not support dynamic value space`
  - fixed by launching a fresh study / sqlite pair under the new timestamped study name
- third blocker resolved during same-session debug:
  - the wrapper temporarily omitted `--model-type ./Llama-3.2-1B`, so the run fell back to the
    gated default `meta-llama/Llama-3.2-1B` and failed remote auth
  - fixed by restoring the explicit local model path in the wrapper and relaunching under tmux
- current live monitoring snapshot:
  - remote tmux window `stage3-exp193-optuna` is up
  - active study remains `exp193_user_ltv_optuna_20260514t141722`
  - remote process chain is alive on GPUs `1,2`
  - remote study directory later advanced through at least `trial_0010.log`
  - verified on 2026-05-14 18:09 CST:
    - live processes: `tune_hyperparameters.py`, `torch.distributed.run`, and two `main.py` ranks
    - current active trial is `trial 10`
    - completed trial count is `8`, with `trial 0` failed and best still `trial 6`
      at `mae ~= 70.44936827350408`
    - latest completed visible trial was `trial 9` with `mae ~= 84.77386943960738`
    - active test-subset throughput was about `26.7` items/sec near the end of the last visible
      subset evaluation
  - the Optuna sqlite also still shows an older `trial 1` in `RUNNING` state from an earlier
    interrupted attempt; this appears to be stale metadata rather than a second live trainer
    because only one real 2-rank process chain is active on the host

## Next required continuation

- `exp193_user_ltv_optuna_20260514t141722` is completed and synced back to the clean worktree
- next step is a separate final-test-only decision for the selected best settings
- do not bundle final full test into any future Optuna rerun

## Phase 2 live monitoring snapshot on 2026-05-14 late evening CST

- verified again from the clean worktree against the real remote host `lab25211`
- remote tmux windows:
  - `0:bash`
  - `1:stage3-exp194-final`
  - `2:stage3-exp192-final`
- live final-test process placement:
  - `exp192` / `user-churn`:
    - wrapper `tune_hyperparameters.py --final-test-only` alive on GPUs `0,1`
    - 2-rank DDP launch via `torch.distributed.run`
    - best setting still uses per-rank `batch_size=3`, so global batch is `6`
    - current sampled log line:
      - `[Test]:   7%|▋         | 12920/175943 [07:26<1:33:43, 28.99it/s]`
    - projected remaining wall clock at sample time:
      - about `1h34m`
    - throughput interpretation:
      - about `14.5` test items/sec/GPU
  - `exp194` / `item-incoterms`:
    - wrapper `tune_hyperparameters.py --final-test-only` alive on GPUs `2,3`
    - 2-rank DDP launch via `torch.distributed.run`
    - best setting still uses per-rank `batch_size=4`, so global batch is `8`
    - current sampled log line:
      - `[Test]:  61%|██████▏   | 123707/201418 [1:51:01<1:09:48, 18.55it/s]`
    - projected remaining wall clock at sample time:
      - about `1h10m`
    - throughput interpretation:
      - about `9.3` test items/sec/GPU
- `nvidia-smi` and process inspection were consistent with only these two Stage 3 final tests using
  GPUs `0-3`
- conclusion:
  - both remaining Phase 2 final-test-only runs are genuinely alive and advancing
  - no relaunch was needed during this monitoring pass
  - next required action stays the same: wait for whichever run finishes first, then immediately
    sync its `best_trial.json`, `best_trial_test.log`, and `/tmp/stage3-exp19x-final.log` back to
    the clean worktree and update the corresponding report

## New-thread revalidation on 2026-05-14 23:15 CST

- takeover was repeated from the clean worktree only:
  - `G:\RelLLM-2\Rel-LLM-clean-p13`
  - branch still `codex/stage3-clean-p13`
- local clean-worktree status was rechecked before touching the remote host:
  - tracked Stage 3 files such as `main.py`, `model.py`, `text_embedder.py`,
    `tune_hyperparameters.py`, and `TUNE_SCRIPT_README.md` are already modified in this worktree
  - Stage 3 directories such as `stage3_notes/`, `optuna_runs/`, `gnn_repr/`, and `precompute/`
    are present as local untracked state in this clean worktree
  - therefore continuation must stay in this worktree and must not fall back to the old dirty repo
- refreshed the thread heartbeat automation instead of trusting the older thread target:
  - id: `monitor-exp192-optuna-continuation`
  - cadence: `30` minutes
  - prompt now explicitly says to re-check tmux, GPU state, processes, logs, and remote
    `optuna_runs`, keep Optuna separate from final full test, sync artifacts immediately after
    completion, and continue the next concrete action rather than stopping after one check
- remote revalidation against the real host `lab25211` found the same two active final-test-only
  runs still alive:
  - tmux windows:
    - `0:bash`
    - `1:stage3-exp194-final`
    - `2:stage3-exp192-final`
  - remote repo used by the live process chain:
    - `/fs/fast/u2021201693/lym/Rel-LLM`
  - `exp192` / `user-churn`:
    - live wrapper `tune_hyperparameters.py --final-test-only` on GPUs `0,1`
    - live `torch.distributed.run` plus two `main.py` ranks
    - sampled progress from `best_trial_test.log`:
      - `63266/175943` at `36:27` elapsed and about `28.6` test examples/sec aggregate
    - throughput interpretation:
      - about `14.3` test examples/sec/GPU
  - `exp194` / `item-incoterms`:
    - live wrapper `tune_hyperparameters.py --final-test-only` on GPUs `2,3`
    - live `torch.distributed.run` plus two `main.py` ranks
    - sampled progress from `best_trial_test.log`:
      - `156531/201418` at `2:20:01` elapsed and about `19.2` test examples/sec aggregate
    - throughput interpretation:
      - about `9.6` test examples/sec/GPU
- artifact state at revalidation time:
  - remote `best_trial.json` and growing `best_trial_test.log` already exist for both
    `exp192_user_churn_optuna_20260513t2307` and `exp194_item_incoterms_optuna_20260513t2307`
  - `exp193_user_ltv_optuna_20260514t141722` still has complete final artifacts on the remote side
- action taken during this pass:
  - no relaunch and no forced interruption, because both remaining Phase 2 runs were verified as
    genuinely alive and advancing
  - next required foreground or heartbeat action remains: when either run finishes, immediately
    sync its final artifacts and update the local scientific record before deciding the next launch

## Phase 2 partial completion on 2026-05-15 00:12 CST

- verified again from the clean worktree against the real remote host `lab25211`
- remote tmux windows had changed to:
  - `0:bash`
  - `2:stage3-exp192-final`
- interpretation:
  - `exp194` / `item-incoterms` final-test-only finished cleanly
  - `exp192` / `user-churn` final-test-only is still alive
- live remaining run:
  - `exp192` / `user-churn`:
    - wrapper plus two DDP ranks still alive on GPUs `0,1`
    - sampled progress from `/tmp/stage3-exp192-final.log`:
      - `163233/175943` at `1:34:19` elapsed and about `28.6-29.3` test examples/sec aggregate
    - throughput interpretation:
      - about `14.3-14.7` test examples/sec/GPU
- completed run now synced locally:
  - `exp194` / `item-incoterms`:
    - synced remote `best_trial.json`
    - synced remote `best_trial_test.log`
    - synced `/tmp/stage3-exp194-final.log`
    - wrote local report:
      - `stage3_notes/reports/exp194_item_incoterms_optuna_20260513t2307.report.json`
  - final full-test metrics:
    - `mrr=0.7254209687717201`
    - `accuracy=0.6087479985602046`
    - `macro_f1=0.09577581222004289`
    - `micro_f1=0.6087479985602046`
  - scientific interpretation:
    - this is better than the stored item-incoterms full-test reference
      (`mrr=0.7043105782857789`, `accuracy=0.580488289249941`)
    - therefore `exp194` has moved from a Phase 1 retune-plausible status to a confirmed positive
      Phase 2 result
- next required continuation:
  - retarget heartbeat monitoring to the still-running `exp192` final-test-only job
  - when `exp192` finishes, sync its artifacts immediately and then judge the combined Phase 2
    outcome before selecting the next launch or document-only blocker

## Phase 2 completion summary on 2026-05-15 00:23 CST

- all three Phase 2 runs now have synchronized local outcomes in the clean worktree:
  - `exp193_user_ltv_optuna_20260514t141722`
  - `exp194_item_incoterms_optuna_20260513t2307`
  - `exp192_user_churn_optuna_20260513t2307`
- `exp192` completion specifics:
  - the remote full test itself finished and printed final metrics:
    - `roc_auc=0.6681287838379838`
    - `average_precision=0.7236622468421053`
    - `accuracy=0.6535089588928201`
    - `f1=0.7253997103662313`
  - however, the `--final-test-only` wrapper stalled before printing the final save lines even
    though `best_trial_test.log` had already reached the completed metrics line
  - after syncing the completed log artifacts, the stale remote process chain and tmux window were
    cleared so GPUs `0,1` would not remain blocked
  - the local clean-worktree copy of `best_trial.json` now contains the recovered `final_test`
    payload from the completed log
- scientific summary across Phase 2:
  - `exp193` / `user-ltv`: full-test improvement over reference
  - `exp194` / `item-incoterms`: full-test improvement over reference
  - `exp192` / `user-churn`: full test remained below reference
- resulting program judgment:
  - the EXP192 mechanism family now has real full-test wins and should not be collapsed back into a
    simple “failed screen” narrative
  - at the same time, this is not a full representative-set confirmation because `user-churn`
    missed its full-test reference
  - the correct persisted state is: mixed but scientifically meaningful Phase 2 outcome, worth
    carrying forward into the next Stage 3 selection step rather than rerunning the same final-test
    wave

## Post-completion revalidation on 2026-05-15

- rechecked the real remote host `lab25211` again from the clean worktree after Phase 2 closure
- remote tmux state is now back to only:
  - `0:bash`
- there is no live remote `tune_hyperparameters.py`, `torch.distributed.run`, or `main.py`
  process chain belonging to the finished EXP192-family Phase 2 runs
- the remote `best_trial.json` and `best_trial_test.log` outputs remain present for:
  - `exp192_user_churn_optuna_20260513t2307`
  - `exp193_user_ltv_optuna_20260514t141722`
  - `exp194_item_incoterms_optuna_20260513t2307`
- conclusion:
  - the old monitoring heartbeat target is now stale
  - the next heartbeat should be retargeted away from final-test monitoring and toward the next
    concrete Part 3 / follow-up-selection blocker
  - the current clean-worktree state is ready to be committed as the validation-backed Part 1 base
    before starting the next follow-up
