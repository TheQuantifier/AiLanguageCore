param(
    [ValidateSet('autotrain', 'improve')]
    [string]$Command = 'autotrain',
    [string]$Type,
    [int]$MaxIterations = 30,
    [int]$NumTrainEpochs,
    [string]$Config,
    [string]$Prompt = 'Finished running train. Analyze the latest completed native run, apply the next improvement so I can run the next train, and stop only if another training iteration is not the right next step.',
    [string]$CodexPath,
    [string]$CodexModel = 'gpt-5.3-codex',
    [double]$RecoveryThreshold = 0.50,
    [switch]$OpenStatusWindow = $true,
    [double]$MinFreeMemoryGB = 2.0,
    [double]$MinFreeDiskGB = 5.0,
    [switch]$StopOnBattery = $true,
    [int]$MinBatteryPercent = 35,
    [int]$SafetyCheckIntervalSeconds = 20,
    [int]$MaxSafetyWaitMinutes = 10,
    [int]$MaxCpuThreads = 0,
    [switch]$LowerPriority = $true
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

function Get-AutotrainSafetySnapshot {
    param(
        [string]$RepoRoot
    )

    $resolvedRoot = (Resolve-Path $RepoRoot).Path
    $os = Get-CimInstance -ClassName Win32_OperatingSystem
    $freeMemoryGb = [double]$os.FreePhysicalMemory / 1MB
    $totalMemoryGb = [double]$os.TotalVisibleMemorySize / 1MB

    $driveRoot = [System.IO.Path]::GetPathRoot($resolvedRoot)
    $driveName = $driveRoot.Substring(0, 1)
    $drive = Get-PSDrive -Name $driveName
    $freeDiskGb = [double]$drive.Free / 1GB

    $battery = $null
    try {
        $battery = Get-CimInstance -ClassName Win32_Battery -ErrorAction Stop | Select-Object -First 1
    } catch {
        $battery = $null
    }

    $hasBattery = $null -ne $battery
    $batteryPercent = $null
    $batteryStatus = $null
    $onBattery = $false
    if ($hasBattery) {
        if ($null -ne $battery.EstimatedChargeRemaining) {
            $batteryPercent = [int]$battery.EstimatedChargeRemaining
        }
        if ($null -ne $battery.BatteryStatus) {
            $batteryStatus = [int]$battery.BatteryStatus
        }
        if ($batteryStatus -in @(1, 4, 5, 11)) {
            $onBattery = $true
        }
    }

    return [pscustomobject]@{
        FreeMemoryGb = $freeMemoryGb
        TotalMemoryGb = $totalMemoryGb
        FreeDiskGb = $freeDiskGb
        DriveName = $driveName
        HasBattery = $hasBattery
        BatteryPercent = $batteryPercent
        BatteryStatus = $batteryStatus
        OnBattery = $onBattery
    }
}

function Test-AutotrainSafety {
    param(
        [object]$Snapshot,
        [double]$RequiredFreeMemoryGb,
        [double]$RequiredFreeDiskGb,
        [bool]$ShouldStopOnBattery,
        [int]$RequiredBatteryPercent
    )

    $issues = @()
    if ($Snapshot.FreeMemoryGb -lt $RequiredFreeMemoryGb) {
        $issues += ("free RAM {0:N2} GB < required {1:N2} GB" -f $Snapshot.FreeMemoryGb, $RequiredFreeMemoryGb)
    }
    if ($Snapshot.FreeDiskGb -lt $RequiredFreeDiskGb) {
        $issues += ("free disk ({0}:) {1:N2} GB < required {2:N2} GB" -f $Snapshot.DriveName, $Snapshot.FreeDiskGb, $RequiredFreeDiskGb)
    }
    if ($ShouldStopOnBattery -and $Snapshot.HasBattery -and $Snapshot.OnBattery) {
        $issues += 'running on battery power'
    }
    if ($ShouldStopOnBattery -and $Snapshot.HasBattery -and $null -ne $Snapshot.BatteryPercent -and $Snapshot.BatteryPercent -lt $RequiredBatteryPercent) {
        $issues += ("battery {0}% < required {1}%" -f $Snapshot.BatteryPercent, $RequiredBatteryPercent)
    }

    return [pscustomobject]@{
        IsSafe = ($issues.Count -eq 0)
        Issues = $issues
    }
}

function Wait-ForAutotrainSafeResources {
    param(
        [string]$RepoRoot,
        [double]$RequiredFreeMemoryGb,
        [double]$RequiredFreeDiskGb,
        [bool]$ShouldStopOnBattery,
        [int]$RequiredBatteryPercent,
        [int]$PollIntervalSeconds,
        [int]$MaxWaitMinutes
    )

    $deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
    while ($true) {
        $snapshot = Get-AutotrainSafetySnapshot -RepoRoot $RepoRoot
        $safety = Test-AutotrainSafety `
            -Snapshot $snapshot `
            -RequiredFreeMemoryGb $RequiredFreeMemoryGb `
            -RequiredFreeDiskGb $RequiredFreeDiskGb `
            -ShouldStopOnBattery $ShouldStopOnBattery `
            -RequiredBatteryPercent $RequiredBatteryPercent

        if ($safety.IsSafe) {
            Write-Host ("Safety check passed | free_ram={0:N2}GB/{1:N2}GB | free_disk={2:N2}GB | on_battery={3}" -f $snapshot.FreeMemoryGb, $snapshot.TotalMemoryGb, $snapshot.FreeDiskGb, $snapshot.OnBattery)
            return $snapshot
        }

        $remaining = [int][Math]::Ceiling(($deadline - (Get-Date)).TotalSeconds)
        if ($remaining -le 0) {
            $issueText = $safety.Issues -join '; '
            throw "Safety guard timeout after $MaxWaitMinutes minute(s): $issueText"
        }
        Write-Warning ("Safety guard waiting ({0}s left): {1}" -f $remaining, ($safety.Issues -join '; '))
        Start-Sleep -Seconds ([Math]::Max(1, $PollIntervalSeconds))
    }
}

function Acquire-AutotrainLock {
    param(
        [string]$LockPath
    )

    try {
        $stream = [System.IO.File]::Open($LockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    } catch {
        throw "Another autotrain process appears to be running (lock held at $LockPath). Stop the other process or remove stale lock after confirming no active run."
    }

    $payload = "pid=$PID`nstarted_at=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))`n"
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
    $stream.SetLength(0)
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Flush()
    return $stream
}

function Release-AutotrainLock {
    param(
        $LockHandle,
        [string]$LockPath
    )

    if ($LockHandle) {
        try {
            $LockHandle.Dispose()
        } catch {
            Write-Warning "Failed to release autotrain lock handle: $($_.Exception.Message)"
        }
    }

    if (Test-Path $LockPath) {
        Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-CodexExecutable {
    param(
        [string]$OverridePath
    )

    if ($OverridePath) {
        return (Resolve-Path $OverridePath).Path
    }

    $command = Get-Command codex -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $defaultPath = Join-Path $env:USERPROFILE '.vscode\extensions\openai.chatgpt-26.325.31654-win32-x64\bin\windows-x86_64\codex.exe'
    if (Test-Path $defaultPath) {
        return (Resolve-Path $defaultPath).Path
    }

    $extensionRoot = Join-Path $env:USERPROFILE '.vscode\extensions'
    if (Test-Path $extensionRoot) {
        $candidate = Get-ChildItem -Path $extensionRoot -Filter codex.exe -Recurse -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) {
            return $candidate.FullName
        }
    }

    throw 'Could not locate codex.exe. Install the Codex CLI or pass -CodexPath.'
}

function Read-AutomationDecision {
    param(
        [string]$MessagePath
    )

    if (-not (Test-Path $MessagePath)) {
        throw "Codex did not write a last-message file: $MessagePath"
    }

    $message = Get-Content $MessagePath -Raw
    $match = [regex]::Match($message, 'AUTOMATION_DECISION:\s*(CONTINUE|STOP)', 'IgnoreCase')
    if (-not $match.Success) {
        throw "Codex response did not include AUTOMATION_DECISION. See $MessagePath"
    }

    return $match.Groups[1].Value.ToUpperInvariant()
}

function Test-AutomationDecisionPresent {
    param(
        [string]$MessagePath
    )

    if (-not (Test-Path $MessagePath)) {
        return $false
    }

    try {
        $message = Get-Content $MessagePath -Raw
    } catch {
        return $false
    }

    return [regex]::IsMatch($message, 'AUTOMATION_DECISION:\s*(CONTINUE|STOP)', 'IgnoreCase')
}

function Show-FilteredCodexOutput {
    param(
        [string[]]$Lines
    )

    $skipPrefixes = @(
        'diff --git ',
        'index ',
        '--- a/',
        '+++ b/',
        '@@ ',
        'warning: in the working copy of ',
        'patch: completed',
        'tokens used'
    )

    foreach ($line in $Lines) {
        if ($null -eq $line) {
            continue
        }

        $trimmed = $line.Trim()
        if ($trimmed -eq '') {
            Write-Host ''
            continue
        }

        if ($trimmed -match '^[\+\-]{3,}$') {
            continue
        }

        $shouldSkip = $false
        foreach ($prefix in $skipPrefixes) {
            if ($trimmed.StartsWith($prefix)) {
                $shouldSkip = $true
                break
            }
        }
        if ($shouldSkip) {
            continue
        }

        if ($trimmed -match '^[\+\-].+' -and -not $trimmed.StartsWith('AUTOMATION_DECISION:')) {
            continue
        }

        if ($trimmed -eq 'apply patch' -or $trimmed -eq 'patch' -or $trimmed -eq 'exec') {
            continue
        }

        if ($trimmed -match '^succeeded in \d+ms:$') {
            continue
        }

        Write-Host $line
    }
}

function Get-LatestCodexActivity {
    param(
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $skipPrefixes = @(
        'diff --git ',
        'index ',
        '--- a/',
        '+++ b/',
        '@@ ',
        'warning: in the working copy of ',
        'patch: completed',
        'tokens used',
        'OpenAI Codex ',
        'workdir:',
        'model:',
        'provider:',
        'approval:',
        'sandbox:',
        'reasoning effort:',
        'reasoning summaries:',
        'session id:',
        '--------',
        'user'
    )

    $candidateFiles = @($stdoutPath, $stderrPath)
    foreach ($path in $candidateFiles) {
        if (-not (Test-Path $path)) {
            continue
        }

        $lines = Get-Content -Path $path -Tail 80 -ErrorAction SilentlyContinue
        for ($index = $lines.Count - 1; $index -ge 0; $index--) {
            $trimmed = $lines[$index].Trim()
            if (-not $trimmed) {
                continue
            }
            if ($trimmed -match '^[\+\-]{3,}$') {
                continue
            }
            if ($trimmed -match '^[\+\-].+' -and -not $trimmed.StartsWith('AUTOMATION_DECISION:')) {
                continue
            }
            if ($trimmed -eq 'apply patch' -or $trimmed -eq 'patch' -or $trimmed -eq 'exec') {
                continue
            }
            if ($trimmed -match '^succeeded in \d+ms:$') {
                continue
            }

            $shouldSkip = $false
            foreach ($prefix in $skipPrefixes) {
                if ($trimmed.StartsWith($prefix)) {
                    $shouldSkip = $true
                    break
                }
            }
            if ($shouldSkip) {
                continue
            }

            if ($trimmed.Length -gt 70) {
                return ($trimmed.Substring(0, 67) + '...')
            }
            return $trimmed
        }
    }

    return 'waiting for first activity...'
}

function Invoke-CodexExec {
    param(
        [string]$CodexExecutable,
        [string]$Model,
        [string]$Prompt,
        [string]$RepoRoot,
        [string]$MessagePath,
        [string]$LogPath,
        [switch]$ShowInlineProgress
    )

    function ConvertTo-CommandLineArgument {
        param(
            [string]$Value
        )

        if ($null -eq $Value) {
            return '""'
        }

        if ($Value -eq '') {
            return '""'
        }

        if ($Value -notmatch '[\s"]') {
            return $Value
        }

        $escaped = $Value -replace '(\\*)"', '$1$1\"'
        $escaped = $escaped -replace '(\\+)$', '$1$1'
        return ('"{0}"' -f $escaped)
    }

    $argumentValues = @(
        'exec',
        $Prompt,
        '-C',
        $RepoRoot,
        '-m',
        $Model,
        '--full-auto',
        '-o',
        $MessagePath
    )

    $quotedArguments = $argumentValues | ForEach-Object { ConvertTo-CommandLineArgument -Value $_ }
    $argumentString = [string]::Join(' ', $quotedArguments)

    $stdoutPath = [System.IO.Path]::ChangeExtension($LogPath, '.stdout.log')
    $stderrPath = [System.IO.Path]::ChangeExtension($LogPath, '.stderr.log')

    if (Test-Path $stdoutPath) {
        Remove-Item -LiteralPath $stdoutPath -Force
    }
    if (Test-Path $stderrPath) {
        Remove-Item -LiteralPath $stderrPath -Force
    }

    if ($ShowInlineProgress) {
        Write-Host ("Codex progress preparing | model={0} | starting process..." -f $Model)
    }

    $process = Start-Process -FilePath $CodexExecutable `
        -ArgumentList $argumentString `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    if ($ShowInlineProgress) {
        $spinnerFrames = @('|', '/', '-', '\')
        $spinnerIndex = 0
        $startedAt = Get-Date
        while (-not $process.HasExited) {
            $elapsed = (Get-Date) - $startedAt
            $frame = $spinnerFrames[$spinnerIndex % $spinnerFrames.Count]
            $phase = if ($elapsed.TotalSeconds -lt 5) { 'starting' } else { 'running' }
            $activity = Get-LatestCodexActivity -StdoutPath $stdoutPath -StderrPath $stderrPath
            $line = "Codex progress $frame phase=$phase model=$Model elapsed=$($elapsed.ToString('hh\:mm\:ss')) | $activity"
            Write-Host ("`r" + $line.PadRight(160)) -NoNewline
            Start-Sleep -Milliseconds 1000
            $spinnerIndex += 1
            $process.Refresh()
        }
        Write-Host ("`r" + (" " * 160)) -NoNewline
        Write-Host ("`rCodex progress complete. Logs saved under experiments\\automation.                          ")
    }

    $process.WaitForExit()
    $process.Refresh()

    $stdoutLines = if (Test-Path $stdoutPath) { Get-Content -Path $stdoutPath } else { @() }
    $stderrLines = if (Test-Path $stderrPath) { Get-Content -Path $stderrPath } else { @() }
    $combined = @($stdoutLines + $stderrLines)

    $combined | Out-File -FilePath $LogPath -Encoding utf8

    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Lines = $combined
    }
}

function Get-LatestRunStatus {
    param(
        [string]$RepoRoot,
        [string]$TypeName
    )

    $latestRunPath = Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $RepoRoot -TypeName $TypeName
    $latestStatusPath = Join-Path $latestRunPath 'training_status.json'

    if (-not $latestStatusPath) {
        throw 'Could not find a training_status.json file under models\runs.'
    }

    return Get-Content $latestStatusPath -Raw | ConvertFrom-Json
}

function Get-BenchmarkMetrics {
    param(
        [string]$BenchmarkReportPath
    )

    if (-not $BenchmarkReportPath) {
        throw 'Training status did not include a benchmark report path.'
    }

    if (-not (Test-Path $BenchmarkReportPath)) {
        throw "Benchmark report not found: $BenchmarkReportPath"
    }

    $report = Get-Content $BenchmarkReportPath -Raw | ConvertFrom-Json
    $validOutputRate = if ($null -ne $report.valid_output_rate) {
        [double]$report.valid_output_rate
    } else {
        [double]$report.valid_json_rate
    }

    $nonemptyOutputCount = 0
    if ($null -ne $report.nonempty_output_count) {
        $nonemptyOutputCount = [int]$report.nonempty_output_count
    }

    $validOutputCount = 0
    if ($null -ne $report.valid_output_count) {
        $validOutputCount = [int]$report.valid_output_count
    } elseif ($null -ne $report.valid_json_count) {
        $validOutputCount = [int]$report.valid_json_count
    }

    $correctResponseTypeCount = 0
    if ($null -ne $report.correct_response_type_count) {
        $correctResponseTypeCount = [int]$report.correct_response_type_count
    } elseif ($null -ne $report.response_type_accuracy -and $null -ne $report.benchmark_size) {
        $correctResponseTypeCount = [int][Math]::Round(([double]$report.response_type_accuracy) * [int]$report.benchmark_size)
    }

    $itemsPerSec = 0.0
    if ($null -ne $report.items_per_sec) {
        $itemsPerSec = [double]$report.items_per_sec
    }

    return [pscustomobject]@{
        Path = $BenchmarkReportPath
        NonemptyOutputCount = $nonemptyOutputCount
        ValidOutputRate = $validOutputRate
        ValidOutputCount = $validOutputCount
        CorrectResponseTypeCount = $correctResponseTypeCount
        ResponseTypeAccuracy = [double]$report.response_type_accuracy
        BenchmarkSize = [int]$report.benchmark_size
        ItemsPerSec = $itemsPerSec
    }
}

function Test-AutotrainTargetReached {
    param(
        [object]$BenchmarkMetrics
    )

    if ($null -eq $BenchmarkMetrics) {
        return $false
    }

    if ($BenchmarkMetrics.BenchmarkSize -le 0) {
        return $false
    }

    return (
        $BenchmarkMetrics.CorrectResponseTypeCount -ge $BenchmarkMetrics.BenchmarkSize -and
        $BenchmarkMetrics.ValidOutputCount -ge $BenchmarkMetrics.BenchmarkSize
    )
}

function Invoke-GitPublish {
    param(
        [string]$RepoRoot,
        [string]$CommitMessage
    )

    $statusLines = git -C $RepoRoot status --short
    if (-not $statusLines) {
        Write-Host 'Git publish: working tree clean, nothing to commit.'
        return $false
    }

    Write-Host 'Git publish: git add .'
    git -C $RepoRoot add .
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Git publish failed at git add . Please push manually."
        return $false
    }

    Write-Host ("Git publish: git commit -m ""{0}""" -f $CommitMessage)
    git -C $RepoRoot commit -m $CommitMessage
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Git publish failed at git commit. Please push manually."
        return $false
    }

    Write-Host 'Git publish: git push'
    git -C $RepoRoot push
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Git publish failed at git push. Please push manually."
        return $false
    }

    Write-Host 'Git publish: completed.'
    return $true
}

function Write-AutotrainStatus {
    param(
        [string]$StatusPath,
        [string]$IterationLabel,
        [string]$Phase,
        [string]$Note = '',
        [string]$Workflow = 'train -> benchmark -> codex -> decision',
        [object]$Metrics = $null,
        [string]$TrainingStatusPath = '',
        [string]$BenchmarkStatusPath = ''
    )

    $stageOrder = @('starting', 'training', 'benchmark_complete', 'codex', 'decision', 'stopped', 'interrupted', 'failed')
    $phaseIndex = [Array]::IndexOf($stageOrder, $Phase)
    if ($phaseIndex -lt 0) {
        $phaseIndex = 0
    }

    $payload = [ordered]@{
        iteration_label = $IterationLabel
        phase = $Phase
        timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
        note = $Note
        workflow = $Workflow
        stage = [ordered]@{
            current = $Phase
            index = $phaseIndex + 1
            total = $stageOrder.Count
            order = $stageOrder
        }
    }

    if ($TrainingStatusPath) {
        $payload.training_status = $TrainingStatusPath
    }

    if ($BenchmarkStatusPath) {
        $payload.benchmark_status = $BenchmarkStatusPath
    }

    if ($null -ne $Metrics) {
        $payload.benchmark_size = $Metrics.BenchmarkSize
        $payload.nonempty_output_count = $Metrics.NonemptyOutputCount
        $payload.valid_output_count = $Metrics.ValidOutputCount
        $payload.correct_response_type_count = $Metrics.CorrectResponseTypeCount
        $payload.items_per_sec = $Metrics.ItemsPerSec
        $payload.valid_output_rate = $Metrics.ValidOutputRate
        $payload.response_type_accuracy = $Metrics.ResponseTypeAccuracy
        $payload.benchmark_report = $Metrics.Path
    }

    $payload | ConvertTo-Json -Depth 4 | Set-Content -Path $StatusPath
}

function Invoke-TrainingRun {
    param(
        [string]$RepoRoot,
        [string]$PythonPath,
        [string]$ConfigPath,
        [int]$NumTrainEpochs,
        [bool]$HasEpochOverride
    )

    Push-Location $RepoRoot
    try {
        if ($HasEpochOverride) {
            & $PythonPath scripts\train_native_model.py --config $ConfigPath --num-train-epochs $NumTrainEpochs
        } else {
            & $PythonPath scripts\train_native_model.py --config $ConfigPath
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-CodexImprovementPass {
    param(
        [string]$IterationLabel,
        [string]$IterationDir,
        [string]$RepoRoot,
        [string]$StatusPath,
        [string]$CodexExecutable,
        [string]$CodexModel,
        [string]$Prompt,
        [double]$RecoveryThreshold,
        [object]$BenchmarkMetrics,
        [string]$LatestTrainingStatusPath = '',
        [string]$LatestBenchmarkStatusPath = '',
        [string]$Workflow = 'train -> benchmark -> codex -> decision',
        [switch]$ShowInlineProgress
    )

    $messagePath = Join-Path $IterationDir 'codex_last_message.txt'
    $decisionPath = Join-Path $IterationDir 'decision.txt'
    $codexLogPath = Join-Path $IterationDir 'codex_exec_output.log'

    Write-Host ''
    Write-Host "=== Iteration ${IterationLabel}: Codex improvement pass ==="

    $recoveryInstructions = ''
    if ($BenchmarkMetrics.ValidOutputRate -lt $RecoveryThreshold) {
        $recoveryInstructions = @"

Recovery mode:
- The latest run fell below the minimum acceptable valid output rate threshold.
- First restore or preserve output reliability before attempting more aggressive experiments.
- Prefer reverting the most likely regression source or making the smallest stabilizing change.
- Do not stack multiple speculative changes in one iteration.
"@
    }

    $codexPrompt = @"
$Prompt

You are running inside the AiLanguageCore repository.
The training command has just completed successfully.
Inspect the latest native run under models/runs and its benchmark report under experiments.
Make the next improvement directly in the repo so the next training iteration is the best next step.
Current training type: $selectedType
Current benchmark summary:
- valid_output_rate: $($BenchmarkMetrics.ValidOutputRate)
- response_type_accuracy: $($BenchmarkMetrics.ResponseTypeAccuracy)
- benchmark_report: $($BenchmarkMetrics.Path)

Rules:
- Work only on the next most important improvement.
- Prioritize improving classification accuracy on the current weak boundaries over changing output formatting.
- Avoid retrying class-balanced loss unless you have new evidence it is the right fix.
- Keep changes narrow and comparable so the next training run is an informative experiment.
- In your final response, summarize the change briefly and list the files changed, but do not print a full diff.
- If training should continue after your change, end your final message with exactly: AUTOMATION_DECISION: CONTINUE
- If the automation loop should stop, end your final message with exactly: AUTOMATION_DECISION: STOP
$recoveryInstructions
"@

    Push-Location $RepoRoot
    try {
        Write-AutotrainStatus -StatusPath $StatusPath -IterationLabel $IterationLabel -Phase 'codex' -Note 'Codex improvement pass running.' -Workflow $Workflow -Metrics $BenchmarkMetrics -TrainingStatusPath $LatestTrainingStatusPath -BenchmarkStatusPath $LatestBenchmarkStatusPath
        $codexResult = Invoke-CodexExec -CodexExecutable $CodexExecutable -Model $CodexModel -Prompt $codexPrompt -RepoRoot $RepoRoot -MessagePath $messagePath -LogPath $codexLogPath -ShowInlineProgress:$ShowInlineProgress
        $codexSucceeded = ($null -ne $codexResult.ExitCode -and [int]$codexResult.ExitCode -eq 0)
        if (-not $codexSucceeded -and (Test-AutomationDecisionPresent -MessagePath $messagePath)) {
            Write-Warning 'Codex exited without a usable exit code, but a valid final message was written. Continuing.'
            $codexSucceeded = $true
        }
        if (-not $codexSucceeded) {
            throw "Codex exec failed with exit code $($codexResult.ExitCode)"
        }
        if (-not $ShowInlineProgress) {
            Show-FilteredCodexOutput -Lines $codexResult.Lines
        }
    } finally {
        Pop-Location
    }

    $decision = Read-AutomationDecision -MessagePath $messagePath
    Set-Content -Path $decisionPath -Value $decision
    Write-Host "Codex decision: $decision"

    $summaryLine = ("Loop {0} | {1}/{1} (100.0%) | nonempty_output={2} | valid_output={3} | correct_type={4} | items_per_sec={5:N2}" -f $IterationLabel, $BenchmarkMetrics.BenchmarkSize, $BenchmarkMetrics.NonemptyOutputCount, $BenchmarkMetrics.ValidOutputCount, $BenchmarkMetrics.CorrectResponseTypeCount, $BenchmarkMetrics.ItemsPerSec)
    Write-AutotrainStatus -StatusPath $StatusPath -IterationLabel $IterationLabel -Phase 'decision' -Note ("Codex decision: {0}" -f $decision) -Workflow $Workflow -Metrics $BenchmarkMetrics -TrainingStatusPath $LatestTrainingStatusPath -BenchmarkStatusPath $LatestBenchmarkStatusPath

    return [pscustomobject]@{
        Decision = $decision
        SummaryLine = $summaryLine
        MessagePath = $messagePath
        DecisionPath = $decisionPath
        LogPath = $codexLogPath
    }
}

$pythonPath = Join-Path $repoRoot '.python\python.exe'
$hasEpochOverride = $PSBoundParameters.ContainsKey('NumTrainEpochs')

# Backward/CLI compatibility: when invoked positionally as
# `autotrain <iterations> <epochs>`, PowerShell can bind those to
# -Type and -MaxIterations for this script shape. Re-map numeric type to
# iterations and treat the second positional value as epoch override.
$parsedTypeAsIterations = 0
if ($Command -eq 'autotrain' -and $Type -and [int]::TryParse([string]$Type, [ref]$parsedTypeAsIterations)) {
    if ($parsedTypeAsIterations -lt 1) {
        throw "Max iterations must be at least 1. Received: $parsedTypeAsIterations"
    }

    if (-not $PSBoundParameters.ContainsKey('NumTrainEpochs') -and $PSBoundParameters.ContainsKey('MaxIterations')) {
        if ($MaxIterations -lt 1) {
            throw "NumTrainEpochs must be at least 1. Received: $MaxIterations"
        }
        $NumTrainEpochs = [int]$MaxIterations
        $hasEpochOverride = $true
    }

    $MaxIterations = [int]$parsedTypeAsIterations
    $Type = $null
}

$selectedType = if ($Type) {
    $commandDefaultName = if ($Command -eq 'improve') { 'improve' } else { 'autotrain' }
    Resolve-AiLanguageCoreRequestedType -RepoRoot $repoRoot -CommandName $commandDefaultName -TypeName $Type -RequireTrainable
} else {
    if ($Command -eq 'improve') {
        Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'improve'
    } else {
        Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'autotrain'
    }
}
$configPath = if ($Config) {
    if ([System.IO.Path]::IsPathRooted($Config)) {
        $Config
    } else {
        Join-Path $repoRoot $Config
    }
} else {
    Resolve-AiLanguageCoreTrainingConfig -RepoRoot $repoRoot -TypeName $selectedType
}
$codexExecutable = Resolve-CodexExecutable -OverridePath $CodexPath
$logRoot = Join-Path $repoRoot 'experiments\automation'
$statusPath = Join-Path $logRoot 'latest_status.json'
$statusWatcherScript = Join-Path $repoRoot 'scripts\show_autotrain_status.ps1'
$lockPath = Join-Path $logRoot 'autotrain.lock'

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

if (-not (Test-Path $configPath)) {
    throw "Training config not found: $configPath"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

if ($LowerPriority) {
    try {
        [System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
    } catch {
        Write-Warning "Could not lower process priority: $($_.Exception.Message)"
    }
}

if ($MaxCpuThreads -gt 0) {
    $threadValue = [string]$MaxCpuThreads
    $env:OMP_NUM_THREADS = $threadValue
    $env:MKL_NUM_THREADS = $threadValue
    $env:OPENBLAS_NUM_THREADS = $threadValue
    $env:NUMEXPR_NUM_THREADS = $threadValue
}

Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonPath"
Write-Host "Type: $selectedType"
Write-Host "Config: $configPath"
Write-Host "Codex: $codexExecutable"
Write-Host "Codex model: $CodexModel"
Write-Host "Automation logs: $logRoot"
Write-Host "Command: $Command"
Write-Host "Max iterations: $MaxIterations"
if ($hasEpochOverride) {
    Write-Host "Epoch override: $NumTrainEpochs"
}
Write-Host "Recovery threshold (valid output rate): $RecoveryThreshold"
Write-Host ("Safety guard: min_free_ram={0:N2}GB | min_free_disk={1:N2}GB | stop_on_battery={2} | min_battery={3}% | wait={4}m | poll={5}s" -f $MinFreeMemoryGB, $MinFreeDiskGB, $StopOnBattery, $MinBatteryPercent, $MaxSafetyWaitMinutes, $SafetyCheckIntervalSeconds)
if ($LowerPriority) {
    Write-Host "Process priority: BelowNormal"
}
if ($MaxCpuThreads -gt 0) {
    Write-Host "CPU thread cap: $MaxCpuThreads"
}

$workflowName = if ($Command -eq 'improve') { 'codex -> decision' } else { 'train -> benchmark -> codex -> decision' }
$startupNote = if ($Command -eq 'improve') { 'Standalone improve initialized.' } else { 'Autotrain initialized.' }

Write-AutotrainStatus -StatusPath $statusPath -IterationLabel '---' -Phase 'starting' -Note $startupNote -Workflow $workflowName

if ($OpenStatusWindow -and (Test-Path $statusWatcherScript)) {
    Start-Process powershell.exe -ArgumentList @(
        '-NoExit',
        '-ExecutionPolicy', 'Bypass',
        '-File', $statusWatcherScript,
        '-StatusFile', $statusPath
    ) | Out-Null
}

$iteration = 0
$lastSummaryLine = $null
$autotrainLock = $null

try {
    $autotrainLock = Acquire-AutotrainLock -LockPath $lockPath

    if ($Command -eq 'improve') {
        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $iterationLabel = 'improve'
        $iterationDir = Join-Path $logRoot "improve_$timestamp"
        New-Item -ItemType Directory -Force -Path $iterationDir | Out-Null

        $latestRunStatus = Get-LatestRunStatus -RepoRoot $repoRoot -TypeName $selectedType
        $latestTrainingStatusPath = ''
        $latestBenchmarkStatusPath = ''
        if ($latestRunStatus.output_dir) {
            $latestTrainingStatusPath = Join-Path $latestRunStatus.output_dir 'training_status.json'
            $latestBenchmarkStatusPath = Join-Path $latestRunStatus.output_dir 'benchmark_status.json'
        }
        $benchmarkMetrics = Get-BenchmarkMetrics -BenchmarkReportPath $latestRunStatus.benchmark_report

        Write-Host ("Latest benchmark: valid_output_rate={0:P2} | response_type_accuracy={1:P2} | size={2}" -f $benchmarkMetrics.ValidOutputRate, $benchmarkMetrics.ResponseTypeAccuracy, $benchmarkMetrics.BenchmarkSize)
        Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'benchmark_complete' -Note 'Latest completed run loaded for standalone improvement.' -Workflow 'codex -> decision' -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath

        $improveResult = Invoke-CodexImprovementPass -IterationLabel $iterationLabel -IterationDir $iterationDir -RepoRoot $repoRoot -StatusPath $statusPath -CodexExecutable $codexExecutable -CodexModel $CodexModel -Prompt $Prompt -RecoveryThreshold $RecoveryThreshold -BenchmarkMetrics $benchmarkMetrics -LatestTrainingStatusPath $latestTrainingStatusPath -LatestBenchmarkStatusPath $latestBenchmarkStatusPath -Workflow 'codex -> decision' -ShowInlineProgress:(-not $OpenStatusWindow)
        $commitMessage = "improve: standalone codex pass $timestamp"
        Release-AutotrainLock -LockHandle $autotrainLock -LockPath $lockPath
        $autotrainLock = $null
        Invoke-GitPublish -RepoRoot $repoRoot -CommitMessage $commitMessage | Out-Null

        Write-Host $improveResult.SummaryLine
        if ($improveResult.Decision -eq 'STOP') {
            Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'stopped' -Note 'Standalone improve completed and requested stop.' -Workflow 'codex -> decision' -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath
        }
        return
    }

    while ($true) {
        $iteration += 1
        if ($MaxIterations -gt 0 -and $iteration -gt $MaxIterations) {
            Write-Host "Reached MaxIterations=$MaxIterations. Stopping."
            Write-AutotrainStatus -StatusPath $statusPath -IterationLabel ('{0:D3}' -f ($iteration - 1)) -Phase 'stopped' -Note "Reached MaxIterations=$MaxIterations." -Workflow $workflowName
            if ($lastSummaryLine) {
                Write-Host $lastSummaryLine
            }
            break
        }

        $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $iterationLabel = '{0:D3}' -f $iteration
        $iterationDir = Join-Path $logRoot "iteration_${iterationLabel}_$timestamp"
        $latestTrainingStatusPath = ''
        $latestBenchmarkStatusPath = ''

        New-Item -ItemType Directory -Force -Path $iterationDir | Out-Null

        Write-Host ''
        Write-Host "=== Iteration ${iterationLabel}: training ==="
        Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'training' -Note 'Waiting for safe system resources.' -Workflow $workflowName
        $null = Wait-ForAutotrainSafeResources `
            -RepoRoot $repoRoot `
            -RequiredFreeMemoryGb $MinFreeMemoryGB `
            -RequiredFreeDiskGb $MinFreeDiskGB `
            -ShouldStopOnBattery $StopOnBattery `
            -RequiredBatteryPercent $MinBatteryPercent `
            -PollIntervalSeconds $SafetyCheckIntervalSeconds `
            -MaxWaitMinutes $MaxSafetyWaitMinutes
        Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'training' -Note 'Training in progress.' -Workflow $workflowName

        Invoke-TrainingRun -RepoRoot $repoRoot -PythonPath $pythonPath -ConfigPath $configPath -NumTrainEpochs $NumTrainEpochs -HasEpochOverride $hasEpochOverride

        $latestRunStatus = Get-LatestRunStatus -RepoRoot $repoRoot -TypeName $selectedType
        if ($latestRunStatus.output_dir) {
            $latestTrainingStatusPath = Join-Path $latestRunStatus.output_dir 'training_status.json'
            $latestBenchmarkStatusPath = Join-Path $latestRunStatus.output_dir 'benchmark_status.json'
        }
        $benchmarkMetrics = Get-BenchmarkMetrics -BenchmarkReportPath $latestRunStatus.benchmark_report

        Write-Host ("Latest benchmark: valid_output_rate={0:P2} | response_type_accuracy={1:P2} | size={2}" -f $benchmarkMetrics.ValidOutputRate, $benchmarkMetrics.ResponseTypeAccuracy, $benchmarkMetrics.BenchmarkSize)
        Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'benchmark_complete' -Note 'Training and benchmark completed.' -Workflow $workflowName -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath

        if (Test-AutotrainTargetReached -BenchmarkMetrics $benchmarkMetrics) {
            $lastSummaryLine = ("Loop {0} | {1}/{1} (100.0%) | nonempty_output={2} | valid_output={3} | correct_type={4} | items_per_sec={5:N2}" -f $iterationLabel, $benchmarkMetrics.BenchmarkSize, $benchmarkMetrics.NonemptyOutputCount, $benchmarkMetrics.ValidOutputCount, $benchmarkMetrics.CorrectResponseTypeCount, $benchmarkMetrics.ItemsPerSec)
            Write-Host 'Reached autotrain target: valid_output and correct_type are both 100%. Ending automation loop.'
            Write-Host $lastSummaryLine
            Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'stopped' -Note 'Reached autotrain target: valid_output and correct_type are both 100%.' -Workflow $workflowName -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath
            break
        }

        $codexPass = Invoke-CodexImprovementPass -IterationLabel $iterationLabel -IterationDir $iterationDir -RepoRoot $repoRoot -StatusPath $statusPath -CodexExecutable $codexExecutable -CodexModel $CodexModel -Prompt $Prompt -RecoveryThreshold $RecoveryThreshold -BenchmarkMetrics $benchmarkMetrics -LatestTrainingStatusPath $latestTrainingStatusPath -LatestBenchmarkStatusPath $latestBenchmarkStatusPath -Workflow $workflowName -ShowInlineProgress:(-not $OpenStatusWindow)
        $commitMessage = "autotrain: iteration $iterationLabel codex update"
        Release-AutotrainLock -LockHandle $autotrainLock -LockPath $lockPath
        $autotrainLock = $null
        Invoke-GitPublish -RepoRoot $repoRoot -CommitMessage $commitMessage | Out-Null
        $autotrainLock = Acquire-AutotrainLock -LockPath $lockPath
        $decision = $codexPass.Decision
        $lastSummaryLine = $codexPass.SummaryLine

        if ($decision -eq 'STOP') {
            Write-Host "Codex requested stop. Ending automation loop."
            Write-Host $lastSummaryLine
            Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'stopped' -Note 'Codex requested stop.' -Workflow $workflowName -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath
            break
        }

        Write-Host $lastSummaryLine
    }
} catch {
    $failedLabel = if ($iteration -gt 0) { '{0:D3}' -f $iteration } else { '---' }
    Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $failedLabel -Phase 'failed' -Note $_.Exception.Message -Workflow $workflowName
    throw
} finally {
    Release-AutotrainLock -LockHandle $autotrainLock -LockPath $lockPath
}
