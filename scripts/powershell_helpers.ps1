$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$AiLanguageCoreRoot = Split-Path -Parent $ScriptDirectory
$candidatePy = @(
    (Join-Path $AiLanguageCoreRoot '.venv\Scripts\python.exe'),
    (Join-Path $AiLanguageCoreRoot '.python\python.exe')
)
$py = $null
foreach ($candidate in $candidatePy) {
    if (-not (Test-Path $candidate)) {
        continue
    }
    try {
        $cudaAvailable = & $candidate -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
        if (($LASTEXITCODE -eq 0) -and ($cudaAvailable | Select-Object -Last 1).Trim() -eq '1') {
            $py = $candidate
            break
        }
    } catch {
        continue
    }
}
if (-not $py) {
    foreach ($candidate in $candidatePy) {
        if (Test-Path $candidate) {
            $py = $candidate
            break
        }
    }
}
if (-not (Test-Path $py)) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $py = $pythonCommand.Path }
}
if (-not (Test-Path $py)) {
    $python3Command = Get-Command python3 -ErrorAction SilentlyContinue
    if ($python3Command) { $py = $python3Command.Path }
}
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
        [Parameter(Position = 1)][string]$category_or_command,
        [Parameter(Position = 2)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\set.ps1 $type $category_or_command $command_name
        } elseif ($PSBoundParameters.ContainsKey('category_or_command')) {
            .\set.ps1 $type $category_or_command
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\set.ps1 $type
        } else {
            .\set.ps1
        }
    } finally {
        Pop-Location
    }
}

function set_type {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\set_type.ps1 $type $command_name
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\set_type.ps1 $type
        } else {
            .\set_type.ps1
        }
    } finally {
        Pop-Location
    }
}

function set_category {
    param(
        [Parameter(Position = 0)][string]$category,
        [Parameter(Position = 1)][string]$command_name
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('command_name')) {
            .\set_category.ps1 $category $command_name
        } elseif ($PSBoundParameters.ContainsKey('category')) {
            .\set_category.ps1 $category
        } else {
            .\set_category.ps1
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
        [Parameter(Position = 1)][object]$category_or_epochs,
        [Parameter(Position = 2)][int]$epochs,
        [string]$category
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($type_or_epochs -is [int] -or $type_or_epochs -is [long]) {
            .\train.ps1 $type_or_epochs
            return
        }

        if ($PSBoundParameters.ContainsKey('category')) {
            if ($PSBoundParameters.ContainsKey('epochs')) {
                .\train.ps1 $type_or_epochs $category $epochs
            } elseif ($PSBoundParameters.ContainsKey('type_or_epochs')) {
                .\train.ps1 $type_or_epochs -Category $category
            } else {
                .\train.ps1 -Category $category
            }
        } elseif ($PSBoundParameters.ContainsKey('epochs')) {
            .\train.ps1 $type_or_epochs $category_or_epochs $epochs
        } elseif ($PSBoundParameters.ContainsKey('type_or_epochs')) {
            if ($PSBoundParameters.ContainsKey('category_or_epochs')) {
                .\train.ps1 $type_or_epochs $category_or_epochs
            } else {
                .\train.ps1 $type_or_epochs
            }
        } else {
            .\train.ps1
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

function chat {
    param(
        [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$prompt,
        [string]$model_path,
        [switch]$raw
    )

    Push-Location $AiLanguageCoreRoot
    try {
        $arguments = @()
        if ($PSBoundParameters.ContainsKey('model_path')) {
            $arguments += @('-ModelPath', $model_path)
        }
        if ($raw) {
            $arguments += '-Raw'
        }
        if ($prompt) {
            $arguments += $prompt
        }
        .\chat.ps1 @arguments
    } finally {
        Pop-Location
    }
}

function benchmark {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$model_path,
        [string]$category
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('category')) {
            if ($PSBoundParameters.ContainsKey('model_path')) {
                .\benchmark.ps1 $type $model_path -Category $category
            } elseif ($PSBoundParameters.ContainsKey('type')) {
                .\benchmark.ps1 $type -Category $category
            } else {
                .\benchmark.ps1 -Category $category
            }
        } elseif ($PSBoundParameters.ContainsKey('model_path')) {
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

function frozen_benchmark {
    param(
        [Parameter(Position = 0)][string]$type,
        [Parameter(Position = 1)][string]$model_path,
        [string]$category
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('category')) {
            if ($PSBoundParameters.ContainsKey('model_path')) {
                .\frozen_benchmark.ps1 $type $model_path -Category $category
            } elseif ($PSBoundParameters.ContainsKey('type')) {
                .\frozen_benchmark.ps1 $type -Category $category
            } else {
                .\frozen_benchmark.ps1 -Category $category
            }
        } elseif ($PSBoundParameters.ContainsKey('model_path')) {
            .\frozen_benchmark.ps1 $type $model_path
        } elseif ($PSBoundParameters.ContainsKey('type')) {
            .\frozen_benchmark.ps1 $type
        } else {
            .\frozen_benchmark.ps1
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
        [Parameter(Position = 2)][int]$epochs,
        [string]$category
    )

    Push-Location $AiLanguageCoreRoot
    try {
        $type = $null
        $parsedIterations = 0
        if ($null -ne $type_or_iterations) {
            if ($type_or_iterations -is [int] -or $type_or_iterations -is [long]) {
                $parsedIterations = [int]$type_or_iterations
                if ($PSBoundParameters.ContainsKey('max_iterations') -and -not $PSBoundParameters.ContainsKey('epochs')) {
                    $epochs = [int]$max_iterations
                }
                $max_iterations = $parsedIterations
            } else {
                if ([int]::TryParse([string]$type_or_iterations, [ref]$parsedIterations)) {
                    if ($PSBoundParameters.ContainsKey('max_iterations') -and -not $PSBoundParameters.ContainsKey('epochs')) {
                        $epochs = [int]$max_iterations
                    }
                    $max_iterations = $parsedIterations
                } else {
                    $type = [string]$type_or_iterations
                }
            }
        }

        if ($PSBoundParameters.ContainsKey('epochs') -or ($null -ne $epochs -and $epochs -gt 0)) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -Category $category -MaxIterations $max_iterations -NumTrainEpochs $epochs
            } else {
                .\scripts\run_autotrain_loop.ps1 -Category $category -MaxIterations $max_iterations -NumTrainEpochs $epochs
            }
        } elseif ($PSBoundParameters.ContainsKey('max_iterations') -or ($null -ne $max_iterations -and $max_iterations -gt 0)) {
            if ($type) {
                .\scripts\run_autotrain_loop.ps1 -Type $type -Category $category -MaxIterations $max_iterations
            } else {
                .\scripts\run_autotrain_loop.ps1 -Category $category -MaxIterations $max_iterations
            }
        } elseif ($type) {
            .\scripts\run_autotrain_loop.ps1 -Type $type -Category $category
        } else {
            .\scripts\run_autotrain_loop.ps1 -Category $category
        }
    } finally {
        Pop-Location
    }
}

function improve {
    param(
        [Parameter(Position = 0)][string]$type,
        [string]$category
    )

    Push-Location $AiLanguageCoreRoot
    try {
        if ($PSBoundParameters.ContainsKey('type')) {
            .\scripts\run_autotrain_loop.ps1 -Command improve -Type $type -Category $category -OpenStatusWindow:$false
        } else {
            .\scripts\run_autotrain_loop.ps1 -Command improve -Category $category -OpenStatusWindow:$false
        }
    } finally {
        Pop-Location
    }
}
