# Stage 3 Notes

This directory stores persistent Stage 3 optimization records.

## Files

- `STAGE3_PROGRAM.md`
  - the research-program version of the stage3 workflow
- `baseline_registry.json`
  - machine-readable strict screening baselines
- `pipeline_config.json`
  - remote launch and directory config for the stage3 helper CLI
- `stage3_orchestrator.py`
  - local SSH-based monitor for remote stage3 bundles
- `experiment_log.md`
  - running experiment log
- `baseline_commands.md`
  - canonical commands used for current baseline checks
- `candidates/`
  - machine-readable candidate bundle specs
- `reports/`
  - parsed verdict outputs and cached logs

## Usage Rules

1. Every Stage 3 experiment must be recorded here.
2. Baseline numbers should be written down before testing a modification.
3. Temporary remote changes can be tested first, but the result must still be logged here.
4. Only changes with a real gain should later be committed and pushed.
5. Every new candidate should declare either supporting papers or prior failed
   ablation evidence before launch.

## Required Fields Per Experiment

- experiment id
- date
- branch
- target component
- hypothesis
- command
- validation metrics
- test-subset metrics
- conclusion
- next action
