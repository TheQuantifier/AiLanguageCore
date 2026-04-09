param(
    [Parameter(Position = 0)]
    [string]$Type
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $repoRoot 'scripts\command_type_helpers.ps1')

if (-not $Type) {
    $currentType = Get-AiLanguageCoreDefaultType -RepoRoot $repoRoot
    Write-Host "Current default type: $currentType"
    Write-Host "Trainable types: $((Get-AiLanguageCoreTrainableTypeList) -join ', ')"
    Write-Host "Benchmark-only explicit types: account, medical, oos_tool"
    exit 0
}

$canonical = Set-AiLanguageCoreDefaultType -RepoRoot $repoRoot -TypeName $Type
Write-Host "Default command type set to: $canonical"
