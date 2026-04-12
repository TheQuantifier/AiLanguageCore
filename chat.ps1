param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$Prompt,
    [string]$ModelPath,
    [switch]$Raw
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

$py = Join-Path $repoRoot '.python\python.exe'
if (-not (Test-Path $py)) {
    throw "Python runtime not found at $py"
}

if (-not $ModelPath) {
    $ModelPath = Get-AiLanguageCoreLatestCompletedRunPath -RepoRoot $repoRoot -TypeName 'core' -CategoryName 'full_response'
}

$arguments = @(
    (Join-Path $repoRoot 'scripts\chat_native_model.py'),
    '--model-path',
    $ModelPath
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
