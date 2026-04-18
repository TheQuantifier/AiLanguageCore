param(
    [Parameter(Position = 0)]
    [object]$TypeOrEpoch,
    [Parameter(Position = 1)]
    [object]$CategoryOrEpoch,
    [Parameter(Position = 2)]
    [int]$Epochs,
    [string]$Category,
    [string]$Config
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')
$pythonPath = Get-AiLanguageCorePythonPath -RepoRoot $repoRoot

$defaultSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot -CommandName 'train'
$Type = $defaultSelection.Type
$ResolvedCategory = if ($Category) { Resolve-AiLanguageCoreCategory -CategoryName $Category } else { $defaultSelection.Category }
$hasEpochOverride = $PSBoundParameters.ContainsKey('Epochs')
if ($null -ne $TypeOrEpoch) {
    $parsedEpoch = 0
    if ($TypeOrEpoch -is [int] -or $TypeOrEpoch -is [long]) {
        $Epochs = [int]$TypeOrEpoch
        $hasEpochOverride = $true
    } elseif ([int]::TryParse([string]$TypeOrEpoch, [ref]$parsedEpoch)) {
        $Epochs = $parsedEpoch
        $hasEpochOverride = $true
    } else {
        $Type = Resolve-AiLanguageCoreType -TypeName ([string]$TypeOrEpoch) -RequireTrainable
        if (-not $PSBoundParameters.ContainsKey('Category')) {
            $ResolvedCategory = Get-AiLanguageCorePreferredCategoryForType -TypeName $Type
        }
    }
}

if ($null -ne $CategoryOrEpoch) {
    $parsedEpoch = 0
    if ($CategoryOrEpoch -is [int] -or $CategoryOrEpoch -is [long]) {
        $Epochs = [int]$CategoryOrEpoch
        $hasEpochOverride = $true
    } elseif ([int]::TryParse([string]$CategoryOrEpoch, [ref]$parsedEpoch)) {
        $Epochs = [int]$parsedEpoch
        $hasEpochOverride = $true
    } else {
        $ResolvedCategory = Resolve-AiLanguageCoreCategory -CategoryName ([string]$CategoryOrEpoch)
    }
}

$selection = Resolve-AiLanguageCoreSelection -RepoRoot $repoRoot -CommandName 'train' -TypeName $Type -CategoryName $ResolvedCategory -RequireTrainable
$Type = $selection.Type
$ResolvedCategory = $selection.Category

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$resolvedConfig = if ($PSBoundParameters.ContainsKey('Config')) {
    $Config
} else {
    Resolve-AiLanguageCoreTrainingConfig -RepoRoot $repoRoot -TypeName $Type -CategoryName $ResolvedCategory
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
Write-Host "Category: $ResolvedCategory"
if ($hasEpochOverride) {
    Write-Host "Epoch override: $Epochs"
}

Push-Location $repoRoot
try {
    if ($hasEpochOverride) {
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
