param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Prompt,
    [string]$ModelPath,
    [switch]$Raw
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')
$py = Get-AiLanguageCorePythonPath -RepoRoot $repoRoot

$responseSystemPrompt = 'Reply with valid JSON using exactly these keys: response_type, response. response_type must be one of DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE. Keep response to one short, well-formed sentence.'
$fullResponseSystemPrompt = 'Reply with valid JSON using exactly these keys: response_type, reason, response. response_type must be one of DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE. Keep reason and response to one short, well-formed sentence each.'
$labelOnlySystemPrompt = 'Reply with exactly one label: DIRECT_ANSWER, CLARIFICATION, TOOL_NEEDED, or OUT_OF_SCOPE.'

if (-not (Test-Path $py)) {
    throw "Python runtime not found at $py"
}

if (-not $ModelPath) {
    $ModelPath = Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $repoRoot -TypeName 'core' -CategoryName 'response'
}

$systemPrompt = $responseSystemPrompt
$trainingConfigPath = Join-Path $ModelPath 'training_config.json'
if (Test-Path $trainingConfigPath) {
    $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
    $trainFile = [string]$trainingConfig.train_file
    $benchmarkFile = [string]$trainingConfig.benchmark_file
    $combined = "$trainFile $benchmarkFile"
    if ($combined -match 'full_response') {
        $systemPrompt = $fullResponseSystemPrompt
    } elseif ($combined -match 'category_prediction') {
        $systemPrompt = $labelOnlySystemPrompt
    }
}

$arguments = @(
    (Join-Path $repoRoot 'scripts\chat_native_model.py'),
    '--model-path',
    $ModelPath,
    '--system-prompt',
    $systemPrompt
)

if ($Raw) {
    $arguments += '--show-raw'
}

if ($Prompt -and $Prompt.Count -gt 0) {
    $arguments += $Prompt
} else {
    $arguments += '--interactive'
}

& $py @arguments
