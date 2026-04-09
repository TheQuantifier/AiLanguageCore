param(
    [int]$Epochs,
    [string]$Config = 'models\configs\v1_native_byte_transformer_config.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $repoRoot $Config
}

if (-not (Test-Path $configPath)) {
    throw "Training config not found: $configPath"
}

Write-Host "Starting native training from $repoRoot"
Write-Host "Config: $configPath"
if ($PSBoundParameters.ContainsKey('Epochs')) {
    Write-Host "Epoch override: $Epochs"
}

Push-Location $repoRoot
try {
    if ($PSBoundParameters.ContainsKey('Epochs')) {
        & $pythonPath scripts\train_native_model.py --config $configPath --num-train-epochs $Epochs
    } else {
        & $pythonPath scripts\train_native_model.py --config $configPath
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
