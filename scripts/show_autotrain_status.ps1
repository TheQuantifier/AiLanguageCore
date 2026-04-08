param(
    [string]$StatusFile = 'experiments\automation\latest_status.json',
    [int]$RefreshMilliseconds = 1000
)

$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$statusPath = if ([System.IO.Path]::IsPathRooted($StatusFile)) {
    $StatusFile
} else {
    Join-Path $repoRoot $StatusFile
}

function Read-JsonFile {
    param(
        [string]$Path
    )

    if (-not $Path) {
        return $null
    }

    if (-not (Test-Path $Path)) {
        return $null
    }

    try {
        return Get-Content $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Resolve-LatestTrainingStatusPath {
    param(
        [string]$RepoRoot
    )

    $latest = Get-ChildItem -Path (Join-Path $RepoRoot 'models\runs') -Filter training_status.json -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($latest) {
        return $latest.FullName
    }

    return $null
}

function Get-Bar {
    param(
        [double]$Ratio,
        [int]$Width = 28,
        [switch]$Indeterminate
    )

    $safeWidth = [Math]::Max(8, $Width)
    if ($Indeterminate) {
        $tick = [int]([DateTime]::UtcNow.Ticks / 10000000)
        $segmentWidth = [Math]::Max(4, [Math]::Floor($safeWidth / 4))
        $travel = [Math]::Max(1, $safeWidth - $segmentWidth)
        $offset = $tick % ($travel + 1)
        $chars = New-Object char[] $safeWidth
        for ($i = 0; $i -lt $safeWidth; $i++) {
            $chars[$i] = '.'
        }
        for ($i = $offset; $i -lt [Math]::Min($safeWidth, $offset + $segmentWidth); $i++) {
            $chars[$i] = '#'
        }
        return ('[' + (-join $chars) + ']')
    }

    $clampedRatio = [Math]::Max(0.0, [Math]::Min(1.0, $Ratio))
    $filled = [int][Math]::Floor($clampedRatio * $safeWidth)
    if ($clampedRatio -ge 1.0) {
        return ('[' + ('#' * $safeWidth) + ']')
    }
    if ($filled -le 0) {
        return ('[' + ('-' * $safeWidth) + ']')
    }

    return ('[' + ('#' * [Math]::Max(0, $filled - 1)) + '>' + ('-' * ($safeWidth - $filled)) + ']')
}

function Format-Percent {
    param(
        [double]$Value
    )

    return ('{0,5:P1}' -f $Value)
}

function Format-BarLine {
    param(
        [string]$Label,
        [double]$Ratio,
        [string]$Details,
        [switch]$Indeterminate
    )

    $bar = if ($Indeterminate) {
        Get-Bar -Indeterminate
    } else {
        Get-Bar -Ratio $Ratio
    }

    $percentText = if ($Indeterminate) { '  live' } else { Format-Percent -Value $Ratio }
    return ('{0,-12} {1} {2}  {3}' -f $Label, $bar, $percentText, $Details)
}

function Get-StageRatio {
    param(
        [string]$Phase
    )

    $map = @{
        starting = 0.05
        training = 0.30
        benchmark_complete = 0.55
        codex = 0.78
        decision = 0.92
        stopped = 1.0
        interrupted = 1.0
        failed = 1.0
    }

    if ($map.ContainsKey($Phase)) {
        return [double]$map[$Phase]
    }

    return 0.0
}

function Write-SectionTitle {
    param(
        [string]$Title
    )

    Write-Host $Title
    Write-Host ('-' * $Title.Length)
}

while ($true) {
    Clear-Host
    Write-Host 'Autotrain Status Mirror'
    Write-Host ''

    $payload = Read-JsonFile -Path $statusPath
    if ($null -eq $payload) {
        Write-Host 'Waiting for autotrain status file...'
        Write-Host $statusPath
        Start-Sleep -Milliseconds $RefreshMilliseconds
        continue
    }

    $trainingStatusPath = if ($payload.training_status) {
        $payload.training_status
    } else {
        Resolve-LatestTrainingStatusPath -RepoRoot $repoRoot
    }
    $trainingStatus = Read-JsonFile -Path $trainingStatusPath

    $benchmarkStatusPath = if ($payload.benchmark_status) {
        $payload.benchmark_status
    } elseif ($trainingStatus -and $trainingStatus.benchmark_status) {
        $trainingStatus.benchmark_status
    } else {
        $null
    }
    $benchmarkStatus = Read-JsonFile -Path $benchmarkStatusPath

    Write-SectionTitle -Title 'Iteration'
    $iterationLabel = if ($payload.iteration_label) { $payload.iteration_label } else { '---' }
    $phase = if ($payload.phase) { $payload.phase } else { 'starting' }
    Write-Host ("Loop: {0}" -f $iterationLabel)
    Write-Host ("Phase: {0}" -f $phase)
    Write-Host (Format-BarLine -Label 'Pipeline' -Ratio (Get-StageRatio -Phase $phase) -Details 'train -> benchmark -> codex -> decision')

    if ($payload.timestamp) {
        Write-Host ("Updated: {0}" -f $payload.timestamp)
    }
    if ($payload.note) {
        Write-Host ("Note: {0}" -f $payload.note)
    }

    Write-Host ''
    Write-SectionTitle -Title 'Processes'

    if ($trainingStatus) {
        $trainStep = if ($null -ne $trainingStatus.global_step) { [int]$trainingStatus.global_step } else { 0 }
        $trainMax = if ($null -ne $trainingStatus.max_steps) { [int]$trainingStatus.max_steps } else { 0 }
        $trainRatio = if ($trainMax -gt 0) { $trainStep / $trainMax } else { 0.0 }
        $epochText = if ($null -ne $trainingStatus.epoch) { ('epoch={0:N2}' -f [double]$trainingStatus.epoch) } else { 'epoch=--' }
        $trainDetails = '{0}/{1} steps | {2} | status={3}' -f $trainStep, $trainMax, $epochText, $trainingStatus.status
        Write-Host (Format-BarLine -Label 'Training' -Ratio $trainRatio -Details $trainDetails)
        if ($trainingStatus.latest_log) {
            $parts = @()
            if ($null -ne $trainingStatus.latest_log.train_loss) {
                $parts += ('train_loss={0:N4}' -f [double]$trainingStatus.latest_log.train_loss)
            }
            if ($null -ne $trainingStatus.latest_log.validation_loss) {
                $parts += ('val_loss={0:N4}' -f [double]$trainingStatus.latest_log.validation_loss)
            }
            if ($parts.Count -gt 0) {
                Write-Host ("             {0}" -f ($parts -join ' | '))
            }
        }
    } else {
        $details = if ($phase -eq 'training') { 'waiting for training status...' } else { 'idle' }
        Write-Host (Format-BarLine -Label 'Training' -Ratio 0.0 -Details $details)
    }

    if ($benchmarkStatus) {
        $benchCurrent = if ($null -ne $benchmarkStatus.current) { [int]$benchmarkStatus.current } else { 0 }
        $benchTotal = if ($null -ne $benchmarkStatus.total) { [int]$benchmarkStatus.total } else { 0 }
        $benchRatio = if ($benchTotal -gt 0) { $benchCurrent / $benchTotal } else { 0.0 }
        $benchValid = if ($null -ne $benchmarkStatus.valid_output_count) { [int]$benchmarkStatus.valid_output_count } else { 0 }
        $benchCorrect = if ($null -ne $benchmarkStatus.correct_response_type_count) { [int]$benchmarkStatus.correct_response_type_count } else { 0 }
        $benchItemsPerSec = if ($null -ne $benchmarkStatus.items_per_sec) { [double]$benchmarkStatus.items_per_sec } else { 0.0 }
        $benchDetails = '{0}/{1} items | valid={2} | correct={3} | {4:N2} items/s | status={5}' -f `
            $benchCurrent, `
            $benchTotal, `
            $benchValid, `
            $benchCorrect, `
            $benchItemsPerSec, `
            $benchmarkStatus.status
        Write-Host (Format-BarLine -Label 'Benchmark' -Ratio $benchRatio -Details $benchDetails)
        if ($null -ne $benchmarkStatus.valid_output_rate -or $null -ne $benchmarkStatus.response_type_accuracy) {
            $benchMeta = @()
            if ($null -ne $benchmarkStatus.valid_output_rate) {
                $benchMeta += ('valid_rate={0}' -f (Format-Percent -Value ([double]$benchmarkStatus.valid_output_rate)))
            }
            if ($null -ne $benchmarkStatus.response_type_accuracy) {
                $benchMeta += ('accuracy={0}' -f (Format-Percent -Value ([double]$benchmarkStatus.response_type_accuracy)))
            }
            if ($benchMeta.Count -gt 0) {
                Write-Host ("             {0}" -f ($benchMeta -join ' | '))
            }
        }
    } else {
        $benchmarkDetails = if ($phase -eq 'training') { 'queued after training' } elseif ($phase -eq 'codex' -or $phase -eq 'decision' -or $phase -eq 'stopped' -or $phase -eq 'interrupted') { 'completed' } else { 'idle' }
        $benchmarkRatio = if ($benchmarkDetails -eq 'completed') { 1.0 } else { 0.0 }
        Write-Host (Format-BarLine -Label 'Benchmark' -Ratio $benchmarkRatio -Details $benchmarkDetails)
    }

    $codexActive = $phase -eq 'codex'
    $codexDetails = switch ($phase) {
        'starting' { 'waiting for training to finish' }
        'training' { 'queued after benchmark' }
        'benchmark_complete' { 'starting next improvement pass' }
        'codex' { 'analyzing latest run and applying the next change' }
        'decision' { 'final response received; reading automation decision' }
        'stopped' { 'completed and requested stop' }
        'interrupted' { 'interrupted' }
        'failed' { 'failed' }
        default { 'idle' }
    }
    if ($codexActive) {
        Write-Host (Format-BarLine -Label 'Codex' -Ratio 0.0 -Details $codexDetails -Indeterminate)
    } else {
        $codexRatio = if ($phase -eq 'decision' -or $phase -eq 'stopped' -or $phase -eq 'interrupted') { 1.0 } elseif ($phase -eq 'benchmark_complete') { 0.15 } else { 0.0 }
        Write-Host (Format-BarLine -Label 'Codex' -Ratio $codexRatio -Details $codexDetails)
    }

    if ($null -ne $payload.benchmark_size) {
        Write-Host ''
        Write-SectionTitle -Title 'Latest Metrics'
        Write-Host ('Benchmark size: {0}' -f [int]$payload.benchmark_size)
        Write-Host ('Nonempty output: {0}' -f [int]$payload.nonempty_output_count)
        Write-Host ('Valid output: {0} ({1})' -f [int]$payload.valid_output_count, (Format-Percent -Value ([double]$payload.valid_output_rate)))
        Write-Host ('Correct type: {0} ({1})' -f [int]$payload.correct_response_type_count, (Format-Percent -Value ([double]$payload.response_type_accuracy)))
        Write-Host ('Items/sec: {0:N2}' -f [double]$payload.items_per_sec)
    }

    Start-Sleep -Milliseconds $RefreshMilliseconds
}
