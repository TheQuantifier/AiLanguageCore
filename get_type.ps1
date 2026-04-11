param(
    [Parameter(Position = 0)]
    [string]$CommandName
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if ($CommandName) {
    $resolved = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    $selection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot -CommandName $resolved
    Write-Host "$resolved defaults: type=$($selection.Type) | category=$($selection.Category)"
    exit 0
}

 $globalSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot
Write-Host "Global defaults: type=$($globalSelection.Type) | category=$($globalSelection.Category)"
Write-Host "Command defaults:"
foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
    $selection = Get-AiLanguageCoreDefaultSelection -RepoRoot $repoRoot -CommandName $name
    Write-Host "  ${name}: type=$($selection.Type) | category=$($selection.Category)"
}
