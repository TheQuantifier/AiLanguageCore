param(
    [Parameter(Position = 0)]
    [string]$Type,
    [Parameter(Position = 1)]
    [string]$ModelPath,
    [string]$Category,
    [string]$OutputReport
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

$currentSelection = Resolve-AiLanguageCoreSelection -RepoRoot $repoRoot -CommandName 'train' -TypeName $Type -CategoryName $Category -RequireTrainable
$resolvedFrozen = Resolve-AiLanguageCoreFrozenBenchmarkSelection -RepoRoot $repoRoot -CurrentTypeName $currentSelection.Type -CurrentCategoryName $currentSelection.Category

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$benchmarkPath = Resolve-AiLanguageCoreBenchmarkFile -RepoRoot $repoRoot -TypeName $resolvedFrozen.Type -CategoryName $resolvedFrozen.Category
if (-not (Test-Path $benchmarkPath)) {
    throw "Frozen benchmark file not found: $benchmarkPath"
}

$resolvedModelPath = if ($ModelPath) {
    if ([System.IO.Path]::IsPathRooted($ModelPath)) {
        $ModelPath
    } else {
        Join-Path $repoRoot $ModelPath
    }
} else {
    Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $repoRoot -TypeName $currentSelection.Type -CategoryName $currentSelection.Category
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "Model path not found: $resolvedModelPath"
}

Write-Host "Evaluating frozen benchmark from $repoRoot"
Write-Host "Current model selection: type=$($currentSelection.Type) | category=$($currentSelection.Category)"
Write-Host "Frozen benchmark selection: type=$($resolvedFrozen.Type) | category=$($resolvedFrozen.Category) | source=$($resolvedFrozen.Source)"
Write-Host "Model: $resolvedModelPath"
Write-Host "Benchmark file: $benchmarkPath"

Push-Location $repoRoot
try {
    $command = @(
        'scripts\evaluate_native_model.py',
        '--model-path', $resolvedModelPath,
        '--benchmark-file', $benchmarkPath
    )

    if ($OutputReport) {
        $resolvedReport = if ([System.IO.Path]::IsPathRooted($OutputReport)) {
            $OutputReport
        } else {
            Join-Path $repoRoot $OutputReport
        }
        $command += @('--output-report', $resolvedReport)
    }

    & $pythonPath @command
    if ($LASTEXITCODE -ne 0) {
        throw "Frozen benchmark failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
