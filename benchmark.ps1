param(
    [Parameter(Position = 0)]
    [string]$Type,
    [Parameter(Position = 1)]
    [string]$ModelPath,
    [string]$OutputReport
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

$resolvedType = if ($Type) {
    Resolve-AiLanguageCoreRequestedType -RepoRoot $repoRoot -CommandName 'benchmark' -TypeName $Type
} else {
    Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'benchmark'
}

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$benchmarkPath = Resolve-AiLanguageCoreBenchmarkFile -RepoRoot $repoRoot -TypeName $resolvedType
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
    $modelType = if ($resolvedType -in @('default', 'stress')) { $resolvedType } else { $null }
    Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $repoRoot -TypeName $modelType
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "Model path not found: $resolvedModelPath"
}

Write-Host "Evaluating native model from $repoRoot"
Write-Host "Type: $resolvedType"
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
} finally {
    Pop-Location
}
