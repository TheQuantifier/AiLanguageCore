param(
    [Parameter(Position = 0)]
    [string]$CommandName
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if ($CommandName) {
    $resolved = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    $commandType = Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName $resolved
    Write-Host "$resolved default type: $commandType"
    exit 0
}

$globalType = Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot
Write-Host "Global default type: $globalType"
Write-Host "Command defaults:"
Write-Host "  train: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'train')"
Write-Host "  benchmark: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'benchmark')"
Write-Host "  autotrain: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'autotrain')"
Write-Host "  improve: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'improve')"
