# Command Help

Run these commands from the repo root:
`C:\Users\jhand\Documents\Github\AiLanguageCore`

Set a reusable Python helper once per PowerShell session:

```powershell
$py = '.\.python\python.exe'
```

Optional: add short PowerShell helpers for common commands:

```powershell
function create_data {
    param(
        [Parameter(Mandatory = $true)][int]$count,
        [int]$batch
    )

    if ($PSBoundParameters.ContainsKey('batch')) {
        & $py scripts\generate_data.py --count $count --batch-size $batch
    } else {
        & $py scripts\generate_data.py --count $count
    }

    & $py scripts\prepare_dataset.py
    & $py scripts\convert_training_data.py
}
function train { & $py scripts\train_native_model.py --config models\configs\v1_native_byte_transformer_config.json }
function train_qwen { & $py qwen\scripts\train_lora.py --config qwen\models\configs\v1_local_baseline_config.json }
function eval_native { & $py scripts\evaluate_native_model.py --model-path $args[0] }
function eval_qwen { & $py qwen\scripts\evaluate_benchmark.py --model-path $args[0] }
function status { .\scripts\show_training_status.ps1 -Watch }
function status_qwen { .\qwen\scripts\show_training_status.ps1 -Watch }
function summarize_qwen { & $py qwen\scripts\summarize_training_runs.py; if (Test-Path .\qwen\experiments\training_runs_summary.csv) { Invoke-Item .\qwen\experiments\training_runs_summary.csv } }
```

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
benchmark automatically after training completes successfully:

```powershell
& $py scripts\train_native_model.py --config models\configs\v1_native_byte_transformer_config.json
```

With the helper above, you can just run:

```powershell
train
```

Legacy Qwen/LoRA training is still available, but isolated under `qwen/`:

```powershell
& $py qwen\scripts\train_lora.py --config qwen\models\configs\v1_local_baseline_config.json
```

With the helper above, you can just run:

```powershell
train_qwen
```

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

Legacy Qwen outputs are now isolated under:
- `qwen/models/runs/`
- `qwen/experiments/`

## 5. Export Results

The default benchmark is already run automatically after training. If you want
to run it again manually against the printed training output directory:

```powershell
& $py scripts\evaluate_native_model.py --model-path <printed_native_run_output_dir>
```

With the helper above, you can just run:

```powershell
eval_native <printed_native_run_output_dir>
```

Run the isolated Qwen benchmark manually:

```powershell
& $py qwen\scripts\evaluate_benchmark.py --model-path <printed_qwen_run_output_dir>
```

With the helper above, you can just run:

```powershell
eval_qwen <printed_qwen_run_output_dir>
```

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

Summarize the isolated Qwen runs:

```powershell
& $py qwen\scripts\summarize_training_runs.py
```

With the helper above, you can just run:

```powershell
summarize_qwen
```

## Full Local Pipeline

```powershell
& $py scripts\generate_data.py --count 50
& $py scripts\prepare_dataset.py
& $py scripts\convert_training_data.py
& $py scripts\train_native_model.py --config models\configs\v1_native_byte_transformer_config.json
```

## Shortcut Functions

Copy this block into PowerShell when you want the shortcuts available for the
current session:

```powershell
$py = '.\.python\python.exe'

function create_data {
    param(
        [Parameter(Mandatory = $true)][int]$count,
        [int]$batch
    )

    if ($PSBoundParameters.ContainsKey('batch')) {
        & $py scripts\generate_data.py --count $count --batch-size $batch
    } else {
        & $py scripts\generate_data.py --count $count
    }

    & $py scripts\prepare_dataset.py
    & $py scripts\convert_training_data.py
}

function train {
    & $py scripts\train_native_model.py --config models\configs\v1_native_byte_transformer_config.json
}

function train_qwen {
    & $py qwen\scripts\train_lora.py --config qwen\models\configs\v1_local_baseline_config.json
}

function eval_native {
    & $py scripts\evaluate_native_model.py --model-path $args[0]
}

function eval_qwen {
    & $py qwen\scripts\evaluate_benchmark.py --model-path $args[0]
}

function status {
    .\scripts\show_training_status.ps1 -Watch
}

function status_qwen {
    .\qwen\scripts\show_training_status.ps1 -Watch
}

function summarize {
    & $py scripts\summarize_training_runs.py
    if (Test-Path .\experiments\training_runs_summary.csv) {
        Invoke-Item .\experiments\training_runs_summary.csv
    }
}

function summarize_qwen {
    & $py qwen\scripts\summarize_training_runs.py
    if (Test-Path .\qwen\experiments\training_runs_summary.csv) {
        Invoke-Item .\qwen\experiments\training_runs_summary.csv
    }
}
```
