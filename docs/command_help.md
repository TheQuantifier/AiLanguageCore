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
function set {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\set.ps1 $type $command_name
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\set.ps1 $type
        } else {
            .\set.ps1
        }
    } finally {
        Pop-Location
    }
}
function get_type {
    param(
        [Parameter(Position = 0)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\get_type.ps1 $command_name
        } else {
            .\get_type.ps1
        }
    } finally {
        Pop-Location
    }
}
function train {
    param(
        [Parameter(Position = 0)][object]$type_or_epochs,
        [Parameter(Position = 1)][int]$epochs
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($type_or_epochs -is [int] -or $type_or_epochs -is [long]) {
            .\train.ps1 $type_or_epochs
            return
        }

        if ($PSBoundParameters.ContainsKey('epochs')) {
            .\train.ps1 $type_or_epochs $epochs
        } elseif ($PSBoundParameters.ContainsKey('type_or_epochs')) {
            .\train.ps1 $type_or_epochs
        } else {
            .\train.ps1
        }
    } finally {
        Pop-Location
    }
}
function improve {
    param(
        [Parameter(Position = 0)][string]$type
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('type')) {
            .\scripts\run_autotrain_loop.ps1 -Command improve -Type $type -OpenStatusWindow:$false
        } else {
            .\scripts\run_autotrain_loop.ps1 -Command improve -OpenStatusWindow:$false
        }
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
function benchmark {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$model_path
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('model_path')) {
            .\benchmark.ps1 $type $model_path
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\benchmark.ps1 $type
        } else {
            .\benchmark.ps1
        }
    } finally {
        Pop-Location
    }
}
function status { Push-Location $AiLanguageCoreRoot; try { .\scripts\show_training_status.ps1 -Watch } finally { Pop-Location } }
function summarize { Push-Location $AiLanguageCoreRoot; try { & $py scripts\summarize_training_runs.py; if (Test-Path .\experiments\training_runs_summary.csv) { Invoke-Item .\experiments\training_runs_summary.csv } } finally { Pop-Location } }
function autotrain {
    param(
        [Parameter(Position = 0)][object]$type_or_iterations,
        [Parameter(Position = 1)][int]$max_iterations,
        [Parameter(Position = 2)][int]$epochs
    )

    Push-Location $AiLanguageCoreRoot
    try {
        $type = $null
        if ($null -ne $type_or_iterations) {
            if ($type_or_iterations -is [int] -or $type_or_iterations -is [long]) {
                $max_iterations = [int]$type_or_iterations
            } else {
                $parsedIterations = 0
                if ([int]::TryParse([string]$type_or_iterations, [ref]$parsedIterations)) {
                    $max_iterations = $parsedIterations
                } else {
                    $type = [string]$type_or_iterations
                }
            }
        }

        if ($PSBoundParameters.ContainsKey('epochs')) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -MaxIterations $max_iterations -NumTrainEpochs $epochs
            } else {
                .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations -NumTrainEpochs $epochs
            }
        } elseif ($PSBoundParameters.ContainsKey('max_iterations')) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -MaxIterations $max_iterations
            } else {
                .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations
            }
        } elseif ($type) {
            .\scripts\run_autotrain_loop.ps1 -Type $type
        } else {
            .\scripts\run_autotrain_loop.ps1
        }
    } finally {
        Pop-Location
    }
}
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
- `data/processed/benchmark_stress_native_sft.jsonl`
- `data/processed/benchmark_stress_v2_native_sft.jsonl`
- `data/processed/benchmark_account_tool_boundary_native_sft.jsonl`
- `data/processed/benchmark_medical_refusal_boundary_native_sft.jsonl`
- `data/processed/benchmark_oos_vs_tool_boundary_native_sft.jsonl`

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

`set <type>` sets all type-aware command defaults (`train`, `benchmark`, `autotrain`, `improve`).
`set <type> <command>` sets only that command's default type. It does not run the command.
`core` is the fixed non-stress type. `default` is a dynamic pointer to each command's saved default.
`stress_v2` is a harder stress track with its own benchmark file and training config.

The helper now:
- changes into the repo root automatically
- supports named training types
- supports epoch-only shorthand for the default type
- starts the native trainer from the correct working directory

Default training baseline:
- `train` defaults to the type saved in `config/command_defaults.json`
- `set` changes the saved default trainable type used by `train`, `benchmark`, `autotrain`, and `improve`
- `stress` uses `models\configs\v1_native_byte_transformer_stress_config.json`
- `stress_v2` uses `models\configs\v1_native_byte_transformer_stress_v2_config.json`
- `default` and `core` use `models\configs\v1_native_byte_transformer_config.json`
- all configs default to `50` epochs
- `train <N>` or `.\train.ps1 <N>` uses the currently set default type with an epoch override

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
benchmark stress
benchmark stress_v2
benchmark default
benchmark account
benchmark stress <printed_native_run_output_dir>
```

Named benchmark types:
- `default`, `core` -> `data/processed/benchmark_sft.jsonl`
- `stress` -> `data/processed/benchmark_stress_native_sft.jsonl`
- `stress_v2` -> `data/processed/benchmark_stress_v2_native_sft.jsonl`
- `account` -> `data/processed/benchmark_account_tool_boundary_native_sft.jsonl`
- `medical` -> `data/processed/benchmark_medical_refusal_boundary_native_sft.jsonl`
- `oos_tool` -> `data/processed/benchmark_oos_vs_tool_boundary_native_sft.jsonl`

Default benchmark baseline:
- `benchmark` defaults to the currently set default trainable type
- `account`, `medical`, and `oos_tool` are benchmark-only explicit types
- use `benchmark default` when you want the original default benchmark

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

Copy this block into PowerShell when you want the shortcuts available for the
current session:

```powershell
$AiLanguageCoreRoot = 'C:\Users\jhand\Documents\Github\AiLanguageCore'
$py = Join-Path $AiLanguageCoreRoot '.python\python.exe'
if (Test-Path Alias:set) { Remove-Item Alias:set -Force }

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

function set {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\set.ps1 $type $command_name
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\set.ps1 $type
        } else {
            .\set.ps1
        }
    } finally {
        Pop-Location
    }
}

function get_type {
    param(
        [Parameter(Position = 0)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\get_type.ps1 $command_name
        } else {
            .\get_type.ps1
        }
    } finally {
        Pop-Location
    }
}

function train {
    param(
        [Parameter(Position = 0)][object]$type_or_epochs,
        [Parameter(Position = 1)][int]$epochs
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($type_or_epochs -is [int] -or $type_or_epochs -is [long]) {
            .\train.ps1 $type_or_epochs
            return
        }

        if ($PSBoundParameters.ContainsKey('epochs')) {
            .\train.ps1 $type_or_epochs $epochs
        } elseif ($PSBoundParameters.ContainsKey('type_or_epochs')) {
            .\train.ps1 $type_or_epochs
        } else {
            .\train.ps1
        }
    } finally {
        Pop-Location
    }
}

function improve {
    param(
        [Parameter(Position = 0)][string]$type
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('type')) {
            .\scripts\run_autotrain_loop.ps1 -Command improve -Type $type -OpenStatusWindow:$false
        } else {
            .\scripts\run_autotrain_loop.ps1 -Command improve -OpenStatusWindow:$false
        }
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

function benchmark {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$model_path
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('model_path')) {
            .\benchmark.ps1 $type $model_path
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\benchmark.ps1 $type
        } else {
            .\benchmark.ps1
        }
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
        [Parameter(Position = 0)][object]$type_or_iterations,
        [Parameter(Position = 1)][int]$max_iterations,
        [int]$epochs
    )

    Push-Location $AiLanguageCoreRoot
    try {
        $type = $null
        if ($null -ne $type_or_iterations) {
            if ($type_or_iterations -is [int] -or $type_or_iterations -is [long]) {
                $max_iterations = [int]$type_or_iterations
            } else {
                $parsedIterations = 0
                if ([int]::TryParse([string]$type_or_iterations, [ref]$parsedIterations)) {
                    $max_iterations = $parsedIterations
                } else {
                    $type = [string]$type_or_iterations
                }
            }
        }

        if ($PSBoundParameters.ContainsKey('epochs')) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -MaxIterations $max_iterations -NumTrainEpochs $epochs
            } else {
                .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations -NumTrainEpochs $epochs
            }
        } elseif ($PSBoundParameters.ContainsKey('max_iterations')) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -MaxIterations $max_iterations
            } else {
                .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations
            }
        } elseif ($type) {
            .\scripts\run_autotrain_loop.ps1 -Type $type
        } else {
            .\scripts\run_autotrain_loop.ps1
        }
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
