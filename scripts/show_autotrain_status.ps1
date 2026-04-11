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
        $tick = [long]([DateTime]::UtcNow.Ticks / 10000000)
        $segmentWidth = [Math]::Max(4, [Math]::Floor($safeWidth / 4))
        $travel = [Math]::Max(1, $safeWidth - $segmentWidth)
        $offset = [int]($tick % ($travel + 1))
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

function Format-Duration {
    param(
        [double]$Seconds
    )

    if ($Seconds -lt 0 -or [double]::IsNaN($Seconds) -or [double]::IsInfinity($Seconds)) {
        return '--:--:--'
    }

    $totalSeconds = [Math]::Max(0, [int][Math]::Round($Seconds))
    $hours = [int][Math]::Floor($totalSeconds / 3600)
    $minutes = [int][Math]::Floor(($totalSeconds % 3600) / 60)
    $secs = [int]($totalSeconds % 60)
    return ('{0:D2}:{1:D2}:{2:D2}' -f $hours, $minutes, $secs)
}

function Get-ElapsedSeconds {
    param(
        [string]$StartedAt,
        [string]$UpdatedAt
    )

    if (-not $StartedAt) {
        return $null
    }

    try {
        $start = [DateTimeOffset]::Parse($StartedAt)
        $end = if ($UpdatedAt) { [DateTimeOffset]::Parse($UpdatedAt) } else { [DateTimeOffset]::UtcNow }
        return [Math]::Max(0.0, ($end - $start).TotalSeconds)
    } catch {
        return $null
    }
}

function Get-EtaSeconds {
    param(
        [double]$Current,
        [double]$Total,
        [double]$Rate
    )

    if ($Total -le 0 -or $Current -lt 0 -or $Rate -le 0) {
        return $null
    }

    $remaining = [Math]::Max(0.0, $Total - $Current)
    return $remaining / $Rate
}

function Get-AbsoluteEta {
    param(
        [double]$EtaSeconds
    )

    if ($null -eq $EtaSeconds) {
        return 'ETA=--'
    }

    $etaTime = (Get-Date).AddSeconds($EtaSeconds)
    return ('ETA={0}' -f $etaTime.ToString('HH:mm:ss'))
}

function Format-OptionalDuration {
    param(
        [object]$Seconds
    )

    if ($null -eq $Seconds) {
        return '--:--:--'
    }

    return Format-Duration -Seconds ([double]$Seconds)
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

function Get-ConsoleWidth {
    try {
        return [Math]::Max(60, [Console]::WindowWidth)
    } catch {
        return 120
    }
}

function Wrap-Sections {
    param(
        [string]$Prefix,
        [string[]]$Sections,
        [int]$Width,
        [string]$ContinuationPrefix = ''
    )

    $safeWidth = [Math]::Max(40, $Width)
    $lines = New-Object System.Collections.Generic.List[string]
    $currentPrefix = $Prefix
    $currentLine = $Prefix

    foreach ($section in ($Sections | Where-Object { $_ -and $_.Trim().Length -gt 0 })) {
        $candidate = if ($currentLine -eq $currentPrefix) {
            $currentLine + $section
        } else {
            $currentLine + ' | ' + $section
        }

        if ($candidate.Length -le $safeWidth -or $currentLine -eq $currentPrefix) {
            $currentLine = $candidate
            continue
        }

        $lines.Add($currentLine)
        $currentPrefix = $ContinuationPrefix
        $currentLine = $currentPrefix + $section
    }

    if ($currentLine) {
        $lines.Add($currentLine)
    }

    return $lines.ToArray()
}

function ConvertTo-Lines {
    param(
        [string]$Text
    )

    if ($null -eq $Text) {
        return @('')
    }

    return @([string]$Text -split "`r?`n")
}

function Add-Lines {
    param(
        [System.Collections.Generic.List[string]]$Target,
        [string[]]$Lines
    )

    foreach ($line in $Lines) {
        $Target.Add([string]$line)
    }
}

function Render-Frame {
    param(
        [string[]]$Lines,
        [int]$PreviousLineCount
    )

    $width = Get-ConsoleWidth
    $safeLines = foreach ($line in $Lines) {
        $text = if ($null -eq $line) { '' } else { [string]$line }
        if ($text.Length -gt $width) {
            $text.Substring(0, $width)
        } else {
            $text.PadRight($width)
        }
    }

    $totalLines = [Math]::Max($safeLines.Count, $PreviousLineCount)

    try {
        [Console]::SetCursorPosition(0, 0)
    } catch {
    }

    for ($i = 0; $i -lt $totalLines; $i++) {
        if ($i -lt $safeLines.Count) {
            Write-Host $safeLines[$i]
        } else {
            Write-Host (' ' * $width)
        }
    }

    return $safeLines.Count
}

try {
    [Console]::CursorVisible = $false
} catch {
}

$renderedLineCount = 0
while ($true) {
    $frameLines = New-Object System.Collections.Generic.List[string]
    $frameLines.Add('Autotrain Status Mirror')
    $frameLines.Add('')

    $payload = Read-JsonFile -Path $statusPath
    if ($null -eq $payload) {
        $frameLines.Add('Waiting for autotrain status file...')
        $frameLines.Add($statusPath)
        $renderedLineCount = Render-Frame -Lines $frameLines.ToArray() -PreviousLineCount $renderedLineCount
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

    $consoleWidth = Get-ConsoleWidth
    Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text 'Iteration')
    Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text ('-' * 'Iteration'.Length))
    $iterationLabel = if ($payload.iteration_label) { $payload.iteration_label } else { '---' }
    $phase = if ($payload.phase) { $payload.phase } else { 'starting' }
    $workflow = if ($payload.workflow) { [string]$payload.workflow } else { 'train -> benchmark -> codex -> decision' }
    $frameLines.Add(("Loop: {0}" -f $iterationLabel))
    $frameLines.Add(("Phase: {0}" -f $phase))
    $frameLines.Add((Format-BarLine -Label 'Pipeline' -Ratio (Get-StageRatio -Phase $phase) -Details $workflow))

    if ($payload.timestamp) {
        $frameLines.Add(("Updated: {0}" -f $payload.timestamp))
    }
    if ($payload.note) {
        $frameLines.Add(("Note: {0}" -f $payload.note))
    }

    $frameLines.Add('')
    Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text 'Processes')
    Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text ('-' * 'Processes'.Length))

    if ($trainingStatus) {
        $trainStep = if ($null -ne $trainingStatus.global_step) { [int]$trainingStatus.global_step } else { 0 }
        $trainMax = if ($null -ne $trainingStatus.max_steps) { [int]$trainingStatus.max_steps } else { 0 }
        $trainRatio = if ($trainMax -gt 0) { $trainStep / $trainMax } else { 0.0 }
        $epochText = if ($null -ne $trainingStatus.epoch) { ('epoch={0:N2}' -f [double]$trainingStatus.epoch) } else { 'epoch=--' }
        $trainElapsed = Get-ElapsedSeconds -StartedAt $trainingStatus.started_at -UpdatedAt $trainingStatus.updated_at
        $trainRate = if ($trainElapsed -and $trainElapsed -gt 0 -and $trainStep -gt 0) { $trainStep / $trainElapsed } else { 0.0 }
        $trainEtaSeconds = Get-EtaSeconds -Current $trainStep -Total $trainMax -Rate $trainRate
        $trainDetails = '{0}/{1} steps | {2} | {3} | {4} | {5} | status={6}' -f `
            $trainStep, `
            $trainMax, `
            $epochText, `
            ('elapsed=' + (Format-OptionalDuration -Seconds $trainElapsed)), `
            ('eta=' + (Format-OptionalDuration -Seconds $trainEtaSeconds)), `
            (Get-AbsoluteEta -EtaSeconds $trainEtaSeconds), `
            $trainingStatus.status
        $frameLines.Add((Format-BarLine -Label 'Training' -Ratio $trainRatio -Details $trainDetails))
        if ($trainingStatus.latest_log) {
            $parts = @()
            if ($null -ne $trainingStatus.latest_log.train_loss) {
                $parts += ('train_loss={0:N4}' -f [double]$trainingStatus.latest_log.train_loss)
            }
            if ($null -ne $trainingStatus.latest_log.validation_loss) {
                $parts += ('val_loss={0:N4}' -f [double]$trainingStatus.latest_log.validation_loss)
            }
            if ($null -ne $trainingStatus.latest_log.train_steps_per_second) {
                $parts += ('steps_per_sec={0:N2}' -f [double]$trainingStatus.latest_log.train_steps_per_second)
            }
            if ($null -ne $trainingStatus.latest_log.train_samples_per_second) {
                $parts += ('samples_per_sec={0:N2}' -f [double]$trainingStatus.latest_log.train_samples_per_second)
            }
            if ($parts.Count -gt 0) {
                $metricPrefix = '             '
                Add-Lines -Target $frameLines -Lines (Wrap-Sections -Prefix $metricPrefix -Sections $parts -Width $consoleWidth -ContinuationPrefix $metricPrefix)
            }
        }
    } else {
        $details = if ($phase -eq 'training') { 'waiting for training status...' } else { 'idle' }
        $frameLines.Add((Format-BarLine -Label 'Training' -Ratio 0.0 -Details $details))
    }

    if ($benchmarkStatus) {
        $benchCurrent = if ($null -ne $benchmarkStatus.current) { [int]$benchmarkStatus.current } else { 0 }
        $benchTotal = if ($null -ne $benchmarkStatus.total) { [int]$benchmarkStatus.total } else { 0 }
        $benchRatio = if ($benchTotal -gt 0) { $benchCurrent / $benchTotal } else { 0.0 }
        $benchValid = if ($null -ne $benchmarkStatus.valid_output_count) { [int]$benchmarkStatus.valid_output_count } else { 0 }
        $benchCorrect = if ($null -ne $benchmarkStatus.correct_response_type_count) { [int]$benchmarkStatus.correct_response_type_count } else { 0 }
        $benchItemsPerSec = if ($null -ne $benchmarkStatus.items_per_sec) { [double]$benchmarkStatus.items_per_sec } else { 0.0 }
        $benchElapsed = Get-ElapsedSeconds -StartedAt $benchmarkStatus.started_at -UpdatedAt $benchmarkStatus.updated_at
        $benchEtaSeconds = Get-EtaSeconds -Current $benchCurrent -Total $benchTotal -Rate $benchItemsPerSec
        $benchDetails = '{0}/{1} items | valid={2} | correct={3} | {4:N2} items/s | {5} | {6} | {7} | status={8}' -f `
            $benchCurrent, `
            $benchTotal, `
            $benchValid, `
            $benchCorrect, `
            $benchItemsPerSec, `
            ('elapsed=' + (Format-OptionalDuration -Seconds $benchElapsed)), `
            ('eta=' + (Format-OptionalDuration -Seconds $benchEtaSeconds)), `
            (Get-AbsoluteEta -EtaSeconds $benchEtaSeconds), `
            $benchmarkStatus.status
        $frameLines.Add((Format-BarLine -Label 'Benchmark' -Ratio $benchRatio -Details $benchDetails))
        if ($null -ne $benchmarkStatus.valid_output_rate -or $null -ne $benchmarkStatus.response_type_accuracy) {
            $benchMeta = @()
            if ($null -ne $benchmarkStatus.valid_output_rate) {
                $benchMeta += ('valid_rate={0}' -f (Format-Percent -Value ([double]$benchmarkStatus.valid_output_rate)))
            }
            if ($null -ne $benchmarkStatus.response_type_accuracy) {
                $benchMeta += ('accuracy={0}' -f (Format-Percent -Value ([double]$benchmarkStatus.response_type_accuracy)))
            }
            if ($benchMeta.Count -gt 0) {
                $metricPrefix = '             '
                Add-Lines -Target $frameLines -Lines (Wrap-Sections -Prefix $metricPrefix -Sections $benchMeta -Width $consoleWidth -ContinuationPrefix $metricPrefix)
            }
        }
    } else {
        $benchmarkDetails = if ($phase -eq 'training') { 'queued after training' } elseif ($phase -eq 'codex' -or $phase -eq 'decision' -or $phase -eq 'stopped' -or $phase -eq 'interrupted') { 'completed' } else { 'idle' }
        $benchmarkRatio = if ($benchmarkDetails -eq 'completed') { 1.0 } else { 0.0 }
        $frameLines.Add((Format-BarLine -Label 'Benchmark' -Ratio $benchmarkRatio -Details $benchmarkDetails))
    }

    $codexActive = $phase -eq 'codex'
    $codexElapsed = Get-ElapsedSeconds -StartedAt $payload.timestamp -UpdatedAt $null
    $codexDetails = switch ($phase) {
        'starting' { 'waiting for training to finish' }
        'training' { 'queued after benchmark' }
        'benchmark_complete' { 'starting next improvement pass' }
        'codex' { 'analyzing latest run and applying the next change | elapsed=' + (Format-OptionalDuration -Seconds $codexElapsed) }
        'decision' { 'final response received; reading automation decision' }
        'stopped' { 'completed and requested stop' }
        'interrupted' { 'interrupted' }
        'failed' { 'failed' }
        default { 'idle' }
    }
    if ($codexActive) {
        $frameLines.Add((Format-BarLine -Label 'Codex' -Ratio 0.0 -Details $codexDetails -Indeterminate))
    } else {
        $codexRatio = if ($phase -eq 'decision' -or $phase -eq 'stopped' -or $phase -eq 'interrupted') { 1.0 } elseif ($phase -eq 'benchmark_complete') { 0.15 } else { 0.0 }
        $frameLines.Add((Format-BarLine -Label 'Codex' -Ratio $codexRatio -Details $codexDetails))
    }

    if ($null -ne $payload.benchmark_size) {
        $frameLines.Add('')
        Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text 'Latest Metrics')
        Add-Lines -Target $frameLines -Lines (ConvertTo-Lines -Text ('-' * 'Latest Metrics'.Length))
        $frameLines.Add('Benchmark size: {0}' -f [int]$payload.benchmark_size)
        $frameLines.Add('Nonempty output: {0}' -f [int]$payload.nonempty_output_count)
        $frameLines.Add('Valid output: {0} ({1})' -f [int]$payload.valid_output_count, (Format-Percent -Value ([double]$payload.valid_output_rate)))
        $frameLines.Add('Correct type: {0} ({1})' -f [int]$payload.correct_response_type_count, (Format-Percent -Value ([double]$payload.response_type_accuracy)))
        $frameLines.Add('Items/sec: {0:N2}' -f [double]$payload.items_per_sec)
    }

    $renderedLineCount = Render-Frame -Lines $frameLines.ToArray() -PreviousLineCount $renderedLineCount
    Start-Sleep -Milliseconds $RefreshMilliseconds
}
