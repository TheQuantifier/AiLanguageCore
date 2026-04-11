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
        [Parameter(Position = 2)][int]$epochs
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
                .\scripts\run_autotrain_loop.ps1 -Type $type -MaxIterations $max_iterations -NumTrainEpochs $epochs
            } else {
                .\scripts\run_autotrain_loop.ps1 -MaxIterations $max_iterations -NumTrainEpochs $epochs
            }
        } elseif ($PSBoundParameters.ContainsKey('max_iterations') -or ($null -ne $max_iterations -and $max_iterations -gt 0)) {
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
