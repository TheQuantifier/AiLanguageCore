param(
    [Parameter(Position = 0)]
    [string]$Category,
    [Parameter(Position = 1)]
    [string]$CommandName
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if (-not $Category) {
    Write-Host "Usage: set_category <category> [train|benchmark|frozen_benchmark|autotrain|improve]"
    exit 0
}

$selection = if ($CommandName) {
    Set-AiLanguageCoreDefaultCategoryOnly -RepoRoot $repoRoot -CategoryName $Category -CommandName $CommandName
} else {
    Set-AiLanguageCoreDefaultCategoryOnly -RepoRoot $repoRoot -CategoryName $Category
}

if ($CommandName) {
    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    Write-Host "Category for '$resolvedCommand' set to: $($selection.Category) | type remains $($selection.Type)"
} else {
    Write-Host "Global category set to: $($selection.Category) | type remains $($selection.Type)"
}
