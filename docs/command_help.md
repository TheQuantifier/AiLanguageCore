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
function train { & $py scripts\train_lora.py --config models\v1_local_baseline_config.json }
function status { .\scripts\show_training_status.ps1 -Watch }
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

## 4. Train The Local Model

Run the CPU baseline trainer. It now runs the default benchmark automatically
after training completes successfully:

```powershell
& $py scripts\train_lora.py --config models\v1_local_baseline_config.json
```

With the helper above, you can just run:

```powershell
train
```

Output:
- a new timestamped run directory under `models/runs/`
- example: `models/runs/v1-qwen2.5-0.5b-lora-cpu-20260406-153000`
- the default benchmark report in `experiments/benchmark_report-<run_dir_name>.json`
- the refreshed CSV summary in `experiments/training_runs_summary.csv`

Useful status command while training:

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
& $py scripts\evaluate_benchmark.py --model-path <printed_run_output_dir>
```

Run the separate stress benchmark against the printed training output directory:

```powershell
& $py scripts\evaluate_benchmark.py --model-path <printed_run_output_dir> --benchmark-file data\processed\benchmark_stress_sft.jsonl
```

Run the focused account-tool boundary benchmark against the printed training output directory:

```powershell
& $py scripts\evaluate_benchmark.py --model-path <printed_run_output_dir> --benchmark-file data\processed\benchmark_account_tool_boundary_sft.jsonl
```

Run the out-of-scope versus tool-needed boundary benchmark against the printed training output directory:

```powershell
& $py scripts\evaluate_benchmark.py --model-path <printed_run_output_dir> --benchmark-file data\processed\benchmark_oos_vs_tool_boundary_sft.jsonl
```

Run the medical-refusal boundary benchmark against the printed training output directory:

```powershell
& $py scripts\evaluate_benchmark.py --model-path <printed_run_output_dir> --benchmark-file data\processed\benchmark_medical_refusal_boundary_sft.jsonl
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

If you want to keep a separate copy of the benchmark report for a specific run:

```powershell
Copy-Item experiments\benchmark_report-<run_dir_name>.json experiments\benchmark_report-local-baseline.json
```

## Full Local Pipeline

```powershell
& $py scripts\generate_data.py --count 50
& $py scripts\prepare_dataset.py
& $py scripts\convert_training_data.py
& $py scripts\train_lora.py --config models\v1_local_baseline_config.json
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
    & $py scripts\train_lora.py --config models\v1_local_baseline_config.json
}

function status {
    .\scripts\show_training_status.ps1 -Watch
}

function summarize {
    & $py scripts\summarize_training_runs.py
    Invoke-Item .\experiments\training_runs_summary.csv
}
```
