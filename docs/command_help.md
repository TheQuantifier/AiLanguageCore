# Command Help

Run these commands from the repo root:
`C:\Users\jhand\Documents\Github\AiLanguageCore`

Set a reusable Python helper once per PowerShell session:

```powershell
$AiLanguageCoreRoot = 'C:\Users\jhand\Documents\Github\AiLanguageCore'
$py = Join-Path $AiLanguageCoreRoot '.python\python.exe'
```

Optional: add short PowerShell helpers for common commands:

```powershell
function create_data {
    param(
        [Parameter(Mandatory = $true)][int]$count,
        [int]$batch
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('batch')) {
            & $py scripts\generate_data.py --count $count --batch-size $batch
        } else {
            & $py scripts\generate_data.py --count $count
        }

        & $py scripts\prepare_dataset.py
        & $py scripts\convert_training_data.py
    } finally {
        Pop-Location
    }
}
function train {
    param(
        [int]$epochs
    )

    $config = 'models\configs\v1_native_byte_transformer_config.json'
    Write-Host "Starting native training from $AiLanguageCoreRoot"
    Write-Host "Config: $config"
    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('epochs')) {
            & $py scripts\train_native_model.py --config $config --num-train-epochs $epochs
        } else {
            & $py scripts\train_native_model.py --config $config
        }
    } finally {
        Pop-Location
    }
}
function improve {
    Push-Location $AiLanguageCoreRoot
    try {
        .\scripts\run_autotrain_loop.ps1 -Command improve
    } finally {
        Pop-Location
    }
}
function eval_native {
    param(
        [Parameter(Mandatory = $true)][string]$model_path
    )

    Write-Host "Evaluating native model: $model_path"
    Push-Location $AiLanguageCoreRoot
    try {
        & $py scripts\evaluate_native_model.py --model-path $model_path
    } finally {
        Pop-Location
    }
}
function status { Push-Location $AiLanguageCoreRoot; try { .\scripts\show_training_status.ps1 -Watch } finally { Pop-Location } }
function summarize { Push-Location $AiLanguageCoreRoot; try { & $py scripts\summarize_training_runs.py; if (Test-Path .\experiments\training_runs_summary.csv) { Invoke-Item .\experiments\training_runs_summary.csv } } finally { Pop-Location } }
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
train 8
```

The helper now:
- changes into the repo root automatically
- prints which config it is about to use
- starts the native trainer from the correct working directory

Default training baseline:
- the config now defaults to `30` epochs
- use `train <N>` or `.\train.ps1 -Epochs <N>` only when you want to override that baseline

Run just the Codex improvement pass against the latest completed training run:

```powershell
.\scripts\run_autotrain_loop.ps1 -Command improve
```

With the helper above, you can just run:

```powershell
improve
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
$AiLanguageCoreRoot = 'C:\Users\jhand\Documents\Github\AiLanguageCore'
$py = Join-Path $AiLanguageCoreRoot '.python\python.exe'

function create_data {
    param(
        [Parameter(Mandatory = $true)][int]$count,
        [int]$batch
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('batch')) {
            & $py scripts\generate_data.py --count $count --batch-size $batch
        } else {
            & $py scripts\generate_data.py --count $count
        }

        & $py scripts\prepare_dataset.py
        & $py scripts\convert_training_data.py
    } finally {
        Pop-Location
    }
}

function train {
    param(
        [int]$epochs
    )

    $config = 'models\configs\v1_native_byte_transformer_config.json'
    Write-Host "Starting native training from $AiLanguageCoreRoot"
    Write-Host "Config: $config"
    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('epochs')) {
            & $py scripts\train_native_model.py --config $config --num-train-epochs $epochs
        } else {
            & $py scripts\train_native_model.py --config $config
        }
    } finally {
        Pop-Location
    }
}

function improve {
    Push-Location $AiLanguageCoreRoot
    try {
        .\scripts\run_autotrain_loop.ps1 -Command improve
    } finally {
        Pop-Location
    }
}

function eval_native {
    param(
        [Parameter(Mandatory = $true)][string]$model_path
    )

    Write-Host "Evaluating native model: $model_path"
    Push-Location $AiLanguageCoreRoot
    try {
        & $py scripts\evaluate_native_model.py --model-path $model_path
    } finally {
        Pop-Location
    }
}

function status {
    Push-Location $AiLanguageCoreRoot
    try {
        .\scripts\show_training_status.ps1 -Watch
    } finally {
        Pop-Location
    }
}

function summarize {
    Push-Location $AiLanguageCoreRoot
    try {
        & $py scripts\summarize_training_runs.py
        if (Test-Path .\experiments\training_runs_summary.csv) {
            Invoke-Item .\experiments\training_runs_summary.csv
        }
    } finally {
        Pop-Location
    }
}

function autotrain {
    param(
        [int]$max_iterations = 50
    )

    Push-Location $AiLanguageCoreRoot
    try {
        .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations
    } finally {
        Pop-Location
    }
}
```

## Automated Train Loop

This loop trains, then runs Codex in unattended mode to make the next change,
then trains again until Codex says stop or an error occurs.

Default behavior:
- runs up to `50` iterations
- stops earlier only on error or when Codex returns `AUTOMATION_DECISION: STOP`
- opens a second PowerShell window that mirrors the current autotrain state with live progress bars for:
  - the iteration pipeline
  - training progress
  - benchmark progress
  - Codex activity

Direct command:

```powershell
.\scripts\run_autotrain_loop.ps1
```

Standalone Codex improvement command:

```powershell
.\scripts\run_autotrain_loop.ps1 -Command improve
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
improve
```

Notes:
- This script uses `codex exec --full-auto`.
- Per-iteration automation logs are written under:
  - `experiments/automation`
- The live status window reads:
  - `experiments/automation/latest_status.json`
- The status window is intended to mirror the current state, not act as a scrolling log.
- Codex must finish its final message with one of:
  - `AUTOMATION_DECISION: CONTINUE`
  - `AUTOMATION_DECISION: STOP`
