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
    Write-Host "Usage: set_type <type> [train|benchmark|frozen_benchmark|autotrain|improve]"
    exit 0
}

$selection = if ($CommandName) {
    Set-AiLanguageCoreDefaultTypeOnly -RepoRoot $repoRoot -TypeName $Type -CommandName $CommandName
} else {
    Set-AiLanguageCoreDefaultTypeOnly -RepoRoot $repoRoot -TypeName $Type
}

if ($CommandName) {
    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    Write-Host "Type for '$resolvedCommand' set to: $($selection.Type) | category remains $($selection.Category)"
} else {
    Write-Host "Global type set to: $($selection.Type) | category now $($selection.Category)"
}
