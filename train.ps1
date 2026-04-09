param(
    [Parameter(Position = 0)]
    [object]$TypeOrEpoch = 'stress',
    [Parameter(Position = 1)]
    [int]$Epochs,
    [string]$Config
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'

function Resolve-NativeTrainingConfig {
    param(
        [string]$SelectedType
    )

    $normalized = $SelectedType.Trim().ToLowerInvariant()
    switch ($normalized) {
        'default' { return 'models\configs\v1_native_byte_transformer_config.json' }
        'core' { return 'models\configs\v1_native_byte_transformer_config.json' }
        'base' { return 'models\configs\v1_native_byte_transformer_config.json' }
        'stress' { return 'models\configs\v1_native_byte_transformer_stress_config.json' }
        default {
            throw "Unknown training type '$SelectedType'. Valid types: default, core, base, stress."
        }
    }
}

$Type = 'stress'
if ($null -ne $TypeOrEpoch) {
    $parsedEpoch = 0
    if ($TypeOrEpoch -is [int] -or $TypeOrEpoch -is [long]) {
        $Epochs = [int]$TypeOrEpoch
    } elseif ([int]::TryParse([string]$TypeOrEpoch, [ref]$parsedEpoch)) {
        $Epochs = $parsedEpoch
    } else {
        $Type = [string]$TypeOrEpoch
    }
}

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$resolvedConfig = if ($PSBoundParameters.ContainsKey('Config')) {
    $Config
} else {
    Resolve-NativeTrainingConfig -SelectedType $Type
}

$configPath = if ([System.IO.Path]::IsPathRooted($resolvedConfig)) {
    $resolvedConfig
} else {
    Join-Path $repoRoot $resolvedConfig
}

if (-not (Test-Path $configPath)) {
    throw "Training config not found: $configPath"
}

Write-Host "Starting native training from $repoRoot"
Write-Host "Config: $configPath"
Write-Host "Type: $Type"
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
