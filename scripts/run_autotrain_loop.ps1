param(
    [ValidateSet('autotrain', 'improve')]
    [string]$Command = 'autotrain',
    [int]$MaxIterations = 50,
    [string]$Config = 'models\configs\v1_native_byte_transformer_config.json',
    [string]$Prompt = 'Finished running train. Analyze the latest completed native run, apply the next improvement so I can run the next train, and stop only if another training iteration is not the right next step.',
    [string]$CodexPath,
    [string]$CodexModel = 'gpt-5.3-codex',
    [double]$RecoveryThreshold = 0.50,
    [switch]$OpenStatusWindow = $true
)

$ErrorActionPreference = 'Stop'

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
        [string]$RepoRoot
    )

    $latestStatusPath = Get-ChildItem -Path (Join-Path $RepoRoot 'models\runs') -Filter training_status.json -Recurse -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName

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
        if ($codexResult.ExitCode -ne 0) {
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

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$pythonPath = Join-Path $repoRoot '.python\python.exe'
$configPath = Join-Path $repoRoot $Config
$codexExecutable = Resolve-CodexExecutable -OverridePath $CodexPath
$logRoot = Join-Path $repoRoot 'experiments\automation'
$statusPath = Join-Path $logRoot 'latest_status.json'
$statusWatcherScript = Join-Path $repoRoot 'scripts\show_autotrain_status.ps1'

if (-not (Test-Path $pythonPath)) {
    throw "Python runtime not found: $pythonPath"
}

if (-not (Test-Path $configPath)) {
    throw "Training config not found: $configPath"
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

Write-Host "Repo root: $repoRoot"
Write-Host "Python: $pythonPath"
Write-Host "Config: $configPath"
Write-Host "Codex: $codexExecutable"
Write-Host "Codex model: $CodexModel"
Write-Host "Automation logs: $logRoot"
Write-Host "Command: $Command"
Write-Host "Max iterations: $MaxIterations"
Write-Host "Recovery threshold (valid output rate): $RecoveryThreshold"

Write-AutotrainStatus -StatusPath $statusPath -IterationLabel '---' -Phase 'starting' -Note 'Autotrain initialized.'

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

if ($Command -eq 'improve') {
    $timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $iterationLabel = 'improve'
    $iterationDir = Join-Path $logRoot "improve_$timestamp"
    New-Item -ItemType Directory -Force -Path $iterationDir | Out-Null

    $latestRunStatus = Get-LatestRunStatus -RepoRoot $repoRoot
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
    Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'training' -Note 'Training in progress.'

    Push-Location $repoRoot
    try {
        & $pythonPath scripts\train_native_model.py --config $configPath
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }

    $latestRunStatus = Get-LatestRunStatus -RepoRoot $repoRoot
    if ($latestRunStatus.output_dir) {
        $latestTrainingStatusPath = Join-Path $latestRunStatus.output_dir 'training_status.json'
        $latestBenchmarkStatusPath = Join-Path $latestRunStatus.output_dir 'benchmark_status.json'
    }
    $benchmarkMetrics = Get-BenchmarkMetrics -BenchmarkReportPath $latestRunStatus.benchmark_report

    Write-Host ("Latest benchmark: valid_output_rate={0:P2} | response_type_accuracy={1:P2} | size={2}" -f $benchmarkMetrics.ValidOutputRate, $benchmarkMetrics.ResponseTypeAccuracy, $benchmarkMetrics.BenchmarkSize)
    Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'benchmark_complete' -Note 'Training and benchmark completed.' -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath

    $codexPass = Invoke-CodexImprovementPass -IterationLabel $iterationLabel -IterationDir $iterationDir -RepoRoot $repoRoot -StatusPath $statusPath -CodexExecutable $codexExecutable -CodexModel $CodexModel -Prompt $Prompt -RecoveryThreshold $RecoveryThreshold -BenchmarkMetrics $benchmarkMetrics -LatestTrainingStatusPath $latestTrainingStatusPath -LatestBenchmarkStatusPath $latestBenchmarkStatusPath -ShowInlineProgress:(-not $OpenStatusWindow)
    $commitMessage = "autotrain: iteration $iterationLabel codex update"
    Invoke-GitPublish -RepoRoot $repoRoot -CommitMessage $commitMessage | Out-Null
    $decision = $codexPass.Decision
    $lastSummaryLine = $codexPass.SummaryLine

    if ($decision -eq 'STOP') {
        Write-Host "Codex requested stop. Ending automation loop."
        Write-Host $lastSummaryLine
        Write-AutotrainStatus -StatusPath $statusPath -IterationLabel $iterationLabel -Phase 'stopped' -Note 'Codex requested stop.' -Metrics $benchmarkMetrics -TrainingStatusPath $latestTrainingStatusPath -BenchmarkStatusPath $latestBenchmarkStatusPath
        break
    }

    Write-Host $lastSummaryLine
}
