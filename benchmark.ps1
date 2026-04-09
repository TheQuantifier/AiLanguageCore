param(
    [Parameter(Position = 0)]
    [string]$Type = 'stress',
    [Parameter(Position = 1)]
    [string]$ModelPath,
    [string]$OutputReport
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'

function Resolve-NativeBenchmarkFile {
    param(
        [string]$SelectedType
    )

    $normalized = $SelectedType.Trim().ToLowerInvariant()
    switch ($normalized) {
        'default' { return 'data\processed\benchmark_sft.jsonl' }
        'core' { return 'data\processed\benchmark_sft.jsonl' }
        'base' { return 'data\processed\benchmark_sft.jsonl' }
        'stress' { return 'data\processed\benchmark_stress_native_sft.jsonl' }
        'account' { return 'data\processed\benchmark_account_tool_boundary_native_sft.jsonl' }
        'medical' { return 'data\processed\benchmark_medical_refusal_boundary_native_sft.jsonl' }
        'oos_tool' { return 'data\processed\benchmark_oos_vs_tool_boundary_native_sft.jsonl' }
        default {
            throw "Unknown benchmark type '$SelectedType'. Valid types: default, core, base, stress, account, medical, oos_tool."
        }
    }
}

function Get-LatestCompletedRunPath {
    param(
        [string]$RunsRoot
    )

    $statusPath = Get-ChildItem -Path $RunsRoot -Filter training_status.json -Recurse -File |
        Sort-Object LastWriteTime -Descending |
        Where-Object {
            try {
                $status = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $runDir = Split-Path -Parent $_.FullName
                (Test-Path (Join-Path $runDir 'model.pt')) -and ($status.global_step -gt 0)
            } catch {
                $false
            }
        } |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $statusPath) {
        throw "Could not find a completed training run under $RunsRoot"
    }

    return Split-Path -Parent $statusPath
}

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

$benchmarkRelativePath = Resolve-NativeBenchmarkFile -SelectedType $Type
$benchmarkPath = Join-Path $repoRoot $benchmarkRelativePath
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
    Get-LatestCompletedRunPath -RunsRoot (Join-Path $repoRoot 'models\runs')
}

if (-not (Test-Path $resolvedModelPath)) {
    throw "Model path not found: $resolvedModelPath"
}

Write-Host "Evaluating native model from $repoRoot"
Write-Host "Type: $Type"
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
