# Command Help

Run these commands from the repo root:
`C:\Users\jhand\Documents\Github\AiLanguageCore`

Spreadsheet-style command catalog:
- `docs/commands_reference.csv`

Set a reusable Python helper once per PowerShell session:

```powershell
$AiLanguageCoreRoot = 'C:\Users\jhand\Documents\Github\AiLanguageCore'
$py = Join-Path $AiLanguageCoreRoot '.python\python.exe'
if (Test-Path Alias:set) { Remove-Item Alias:set -Force }
```

Optional: add short PowerShell helpers for common commands:

Preferred source:
- `scripts/powershell_helpers.ps1`
- Load it in PowerShell with: `. .\scripts\powershell_helpers.ps1`

Available helper functions:
- `create_data <count> [batch]`: generate teacher data, prepare the dataset, and rebuild SFT files.
- `set <type> [category] [command]`: save defaults globally or for one command.
- `set_type <type> [category] [command]`: alias for `set`.
- `get_type [command]`: show the saved default selection.
- `train [type] [category] [epochs]`: train using the saved or explicit selection.
- `benchmark [type] [category] [model_path]`: run the active benchmark for the saved or explicit selection.
- `frozen_benchmark [type] [category] [model_path]`: run the regression benchmark for the current model selection.
- `improve [type] [category]`: run the standalone Codex improvement pass.
- `autotrain [type] [category] [max_iterations] [epochs]`: run the automated train -> benchmark -> Codex loop.
- `eval_native <model_path>`: evaluate a saved native model directly.
- `status`: watch live training status.
- `summarize`: refresh and open the training summary CSV.

## 1. Collect AI Training Data

Set your teacher API key first. The generator uses the same `TEACHER_API_KEY`
to generate complete training records directly in each batch.

Example `.env` entry:

```text
TEACHER_API_KEY=your_api_key_here
```

Generate raw teacher examples:

```powershell
& $py scripts\generate_data.py --count 50
```

Output:
- `data/raw/generated_dataset.json`

## 2. Prepare The Dataset

Clean, deduplicate, and split the seed and generated records:

```powershell
& $py scripts\prepare_dataset.py
```

Outputs:
- `data/processed/full_dataset.json`
- `data/processed/train.json`
- `data/processed/validation.json`
- `data/processed/benchmark.json`
- `data/processed/preparation_report.json`

## 3. Convert To Training Data

Convert the processed splits into chat-format JSONL files for SFT:

```powershell
& $py scripts\convert_training_data.py
```

Outputs:
- `data/processed/train_sft.jsonl`
- `data/processed/validation_sft.jsonl`
- `data/processed/benchmark_sft.jsonl`
- `data/processed/benchmark_stress_native_sft.jsonl`
- `data/processed/benchmark_stress_v2_native_sft.jsonl`
- `data/processed/benchmark_account_tool_boundary_native_sft.jsonl`
- `data/processed/benchmark_medical_refusal_boundary_native_sft.jsonl`
- `data/processed/benchmark_oos_vs_tool_boundary_native_sft.jsonl`

Format notes:
- `train_sft.jsonl`, `validation_sft.jsonl`, and `benchmark_sft.jsonl` now use the full V1 response schema with JSON assistant targets: `response_type`, `reason`, and `response`
- the focused native benchmark files such as `benchmark_stress_v2_native_sft.jsonl` remain label-only so they can continue acting as routing and boundary regression tests

## 3a. Data Pipeline In One Command

Run generation, preparation, and conversion in one command:

```powershell
& $py scripts\generate_data.py --count 50; & $py scripts\prepare_dataset.py; & $py scripts\convert_training_data.py
```

With the helper above, you can just run:

```powershell
create_data 50
create_data 200 25
```

This updates:
- `data/raw/generated_dataset.json`
- `data/processed/train.json`
- `data/processed/validation.json`
- `data/processed/benchmark.json`
- `data/processed/train_sft.jsonl`
- `data/processed/validation_sft.jsonl`
- `data/processed/benchmark_sft.jsonl`

## 4. Train Your Native Model

Train the native decoder-only transformer from scratch. It now runs the default
benchmark for the selected type automatically after training completes successfully:

```powershell
.\train.ps1
.\train.ps1 stress
.\train.ps1 stress_v2
.\train.ps1 default
.\train.ps1 stress 8
```

With the helper above, you can just run:

```powershell
set
get_type
get_type train
set stress
set stress_v2
set core
set stress train
train
train stress
train stress_v2
train default
train 8
train stress 8
```

`set <type> [category]` sets all type-aware command defaults.
`set <type> [category] <command>` sets only that command's default selection. It does not run the command.
`set_type` is an alias for the same behavior.
`core` is the fixed non-stress type. `default` is a dynamic pointer to each command's saved default.
`stress_v2` is a harder stress track with its own benchmark file and training config.
`stress_v2` is now frozen as a regression benchmark, so the active defaults should stay on `core` or another non-frozen trainable type.

Categories:
- `category_prediction`
- `full_response`

Examples:
- `set core full_response`
- `set stress category_prediction`
- `set stress_v2 category_prediction frozen_benchmark`

The helper now:
- changes into the repo root automatically
- supports named training types
- supports epoch-only shorthand for the default type
- starts the native trainer from the correct working directory

Default training baseline:
- `train` defaults to `core full_response`
- `benchmark` defaults to the active training selection
- `frozen_benchmark` defaults to `stress_v2 category_prediction`
- `autotrain` and `improve` default to `core`
- `set` changes the saved default trainable type used by `train`, `benchmark`, `autotrain`, and `improve`
- `stress` uses `models\configs\v1_native_byte_transformer_stress_config.json`
- `stress_v2` uses `models\configs\v1_native_byte_transformer_stress_v2_config.json`
- `core category_prediction` uses `models\configs\v1_native_byte_transformer_category_prediction_config.json`
- `core full_response` uses `models\configs\v1_native_byte_transformer_config.json`
- all configs default to `50` epochs
- `train <N>` or `.\train.ps1 <N>` uses the currently set default type with an epoch override
- `autotrain` and `improve` reject frozen types such as `stress_v2`

Run just the Codex improvement pass against the latest completed training run:

```powershell
.\scripts\run_autotrain_loop.ps1 -Command improve
```

With the helper above, you can just run:

```powershell
improve
improve stress
```

This command:
- loads the latest completed native run under `models/runs/`
- reads its benchmark report
- runs the same Codex improvement step used by autotrain
- writes logs under `experiments/automation/improve_<timestamp>/`

Output:
- a new timestamped run directory under `models/runs/`
- example: `models/runs/v1-native-byte-transformer-20260407-180000`
- the default benchmark report in `experiments/benchmark_report-<run_dir_name>.json`
- the refreshed CSV summary in `experiments/training_runs_summary.csv`

Watch native training status:

```powershell
.\scripts\show_training_status.ps1 -Watch
```

With the helper above, you can just run:

```powershell
status
```

## 5. Export Results

The default benchmark is already run automatically after training. If you want
to run it again manually against the printed training output directory:

```powershell
& $py scripts\evaluate_native_model.py --model-path <printed_native_run_output_dir>
```

With the helper above, you can just run:

```powershell
eval_native <printed_native_run_output_dir>
benchmark
benchmark core full_response
benchmark stress_v2 category_prediction
frozen_benchmark
benchmark stress
benchmark stress_v2
benchmark default
benchmark account
benchmark stress <printed_native_run_output_dir>
```

Named benchmark types:
- `core full_response` -> `data/processed/benchmark_full_response_sft.jsonl`
- `core category_prediction` -> `data/processed/benchmark_category_prediction_sft.jsonl`
- `stress category_prediction` -> `data/processed/benchmark_stress_native_sft.jsonl`
- `stress_v2 category_prediction` -> `data/processed/benchmark_stress_v2_native_sft.jsonl`
- `account` -> `data/processed/benchmark_account_tool_boundary_native_sft.jsonl`
- `medical` -> `data/processed/benchmark_medical_refusal_boundary_native_sft.jsonl`
- `oos_tool` -> `data/processed/benchmark_oos_vs_tool_boundary_native_sft.jsonl`

Default benchmark baseline:
- `benchmark` defaults to the saved active selection for the `benchmark` command
- `frozen_benchmark` is the dedicated regression gate
- `account`, `medical`, and `oos_tool` are benchmark-only explicit types
- use `frozen_benchmark` when you want regression coverage instead of the active benchmark

Frozen benchmark fallback order:
- if the current selection is itself frozen and supports that category, use it
- otherwise, use a frozen benchmark in the current category if one exists
- otherwise, fall back to the saved `frozen_benchmark` default

Show a table of all training runs and the currently saved benchmark report for each run:

```powershell
& $py scripts\summarize_training_runs.py
```

This also refreshes:
- `experiments/training_runs_summary.csv`

Show the same summary as JSON:

```powershell
& $py scripts\summarize_training_runs.py --json
```

Outputs:
- `experiments/benchmark_report-<run_dir_name>.json`
- `experiments/training_runs_summary.csv`

## Full Local Pipeline

```powershell
& $py scripts\generate_data.py --count 50
& $py scripts\prepare_dataset.py
& $py scripts\convert_training_data.py
.\train.ps1 stress
.\train.ps1 stress_v2
```

## Shortcut Functions

Load the current helper set from `scripts/powershell_helpers.ps1` when you want
the shortcuts available for the session. The available helpers are:
- `create_data`
- `set`
- `set_type`
- `get_type`
- `train`
- `benchmark`
- `frozen_benchmark`
- `improve`
- `autotrain`
- `eval_native`
- `status`
- `summarize`

## Automated Train Loop

This loop trains, then runs Codex in unattended mode to make the next change,
then trains again until Codex says stop or an error occurs.

Default behavior:
- runs up to `30` iterations
- stops earlier on error, when `correct_type` reaches 100% (`correct_response_type_count == benchmark_size`), or when Codex returns `AUTOMATION_DECISION: STOP`
- opens a second PowerShell window that mirrors the current autotrain state with live progress bars for:
  - the iteration pipeline
  - training progress
  - benchmark progress
  - Codex activity

Direct command:

```powershell
.\scripts\run_autotrain_loop.ps1
.\scripts\run_autotrain_loop.ps1 -Type stress
.\scripts\run_autotrain_loop.ps1 -Type stress_v2
```

Standalone Codex improvement command:

```powershell
.\scripts\run_autotrain_loop.ps1 -Command improve
.\scripts\run_autotrain_loop.ps1 -Command improve -Type stress
.\scripts\run_autotrain_loop.ps1 -Command improve -Type stress_v2
```

Stop after a fixed number of iterations:

```powershell
.\scripts\run_autotrain_loop.ps1 -MaxIterations 3
```

Run without opening the separate status window:

```powershell
.\scripts\run_autotrain_loop.ps1 -OpenStatusWindow:$false
```

With the helper block above loaded:

```powershell
autotrain
autotrain 3
autotrain stress
autotrain stress_v2
autotrain stress 3
autotrain 10 50
improve
improve stress
improve stress_v2
```

Notes:
- This script uses `codex exec --full-auto`.
- `improve` stays in the current window by default.
- Per-iteration automation logs are written under:
  - `experiments/automation`
- The live status window reads:
  - `experiments/automation/latest_status.json`
- The status window is intended to mirror the current state, not act as a scrolling log.
- Codex must finish its final message with one of:
  - `AUTOMATION_DECISION: CONTINUE`
  - `AUTOMATION_DECISION: STOP`
