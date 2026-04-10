param(
    [Parameter(Position = 0)]
    [string]$Type,
    [Parameter(Position = 1)]
    [string]$CommandName
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if (-not $Type) {
    $globalType = Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot
    Write-Host "Global default type: $globalType"
    Write-Host "Command defaults:"
    Write-Host "  train: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'train')"
    Write-Host "  benchmark: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'benchmark')"
    Write-Host "  autotrain: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'autotrain')"
    Write-Host "  improve: $(Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot -CommandName 'improve')"
    Write-Host "Trainable types: default (dynamic), core, stress, stress_v2"
    Write-Host "Concrete types for set: core, stress, stress_v2"
    Write-Host "Benchmark-only explicit types: account, medical, oos_tool"
    Write-Host "Usage: set <type> [train|benchmark|autotrain|improve]"
    exit 0
}

$canonical = Set-AiLanguageCoreDefaultType -RepoRoot $repoRoot -TypeName $Type -CommandName $CommandName
if ($CommandName) {
    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    Write-Host "Default type for '$resolvedCommand' set to: $canonical"
} else {
    Write-Host "Global default type set to: $canonical"
}
