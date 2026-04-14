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

$selection = Resolve-AiLanguageCoreSelection -RepoRoot $repoRoot -CommandName 'benchmark' -TypeName $Type -CategoryName $Category
$resolvedType = $selection.Type
$resolvedCategory = $selection.Category

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$benchmarkPath = Resolve-AiLanguageCoreBenchmarkFile -RepoRoot $repoRoot -TypeName $resolvedType -CategoryName $resolvedCategory
if (-not (Test-Path $benchmarkPath)) {
    throw "Benchmark file not found: $benchmarkPath"
}

$resolvedModelPath = if ($ModelPath) {
    if ([System.IO.Path]::IsPathRooted($ModelPath)) {
        $ModelPath
    } else {
        Join-Path $repoRoot $ModelPath
    }
} else {
    Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $repoRoot -TypeName $resolvedType -CategoryName $resolvedCategory
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "Model path not found: $resolvedModelPath"
}

Write-Host "Evaluating native model from $repoRoot"
Write-Host "Type: $resolvedType"
Write-Host "Category: $resolvedCategory"
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
        throw "Benchmark failed with exit code $LASTEXITCODE"
    }

    & $pythonPath 'scripts\summarize_training_runs.py' '--apply-retention-cleanup'
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark summary cleanup failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
