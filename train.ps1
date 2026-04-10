param(
    [Parameter(Position = 0)]
    [object]$TypeOrEpoch,
    [Parameter(Position = 1)]
    [int]$Epochs,
    [string]$Config
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

$Type = Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'train'
if ($null -ne $TypeOrEpoch) {
    $parsedEpoch = 0
    if ($TypeOrEpoch -is [int] -or $TypeOrEpoch -is [long]) {
        $Epochs = [int]$TypeOrEpoch
    } elseif ([int]::TryParse([string]$TypeOrEpoch, [ref]$parsedEpoch)) {
        $Epochs = $parsedEpoch
    } else {
        $Type = Resolve-AiLanguageCoreRequestedType -RepoRoot $repoRoot -CommandName 'train' -TypeName ([string]$TypeOrEpoch) -RequireTrainable
    }
}

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$resolvedConfig = if ($PSBoundParameters.ContainsKey('Config')) {
    $Config
} else {
    Resolve-AiLanguageCoreTrainingConfig -RepoRoot $repoRoot -TypeName $Type
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
