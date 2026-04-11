param(
    [Parameter(Position = 0)]
    [string]$Type,
    [Parameter(Position = 1)]
    [string]$CategoryOrCommand,
    [Parameter(Position = 2)]
    [string]$CommandName
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if (-not $Type) {
    $globalSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot
    Write-Host "Global defaults: type=$($globalSelection.Type) | category=$($globalSelection.Category)"
    Write-Host "Command defaults:"
    foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
        $selection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot -CommandName $name
        Write-Host "  ${name}: type=$($selection.Type) | category=$($selection.Category)"
    }
    Write-Host "Trainable types: default (dynamic), core, stress, stress_v2"
    Write-Host "Categories: default (dynamic), category_prediction, full_response"
    Write-Host "Concrete types for set: core, stress, stress_v2"
    Write-Host "Benchmark-only explicit types: account, medical, oos_tool"
    Write-Host "Usage: set <type> [category] [train|benchmark|autotrain|improve]"
    exit 0
}

$resolvedCategory = $null
$resolvedCommand = $null
if ($CategoryOrCommand) {
    try {
        $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CategoryOrCommand
    } catch {
        $resolvedCategory = Resolve-AiLanguageCoreCategory -CategoryName $CategoryOrCommand
    }
}
if ($CommandName) {
    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
}

$selection = Set-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot -TypeName $Type -CategoryName $resolvedCategory -CommandName $resolvedCommand
if ($resolvedCommand) {
    Write-Host "Defaults for '$resolvedCommand' set to: type=$($selection.Type) | category=$($selection.Category)"
} else {
    Write-Host "Global defaults set to: type=$($selection.Type) | category=$($selection.Category)"
}
