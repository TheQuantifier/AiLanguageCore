function Get-AiLanguageCoreCommandTypeCatalog {
    return @{
        default = @{
            aliases = @('default', 'core', 'base')
            trainable = $true
            config = 'models\configs\v1_native_byte_transformer_config.json'
            benchmark = 'data\processed\benchmark_sft.jsonl'
        }
        stress = @{
            aliases = @('stress')
            trainable = $true
            config = 'models\configs\v1_native_byte_transformer_stress_config.json'
            benchmark = 'data\processed\benchmark_stress_native_sft.jsonl'
        }
        account = @{
            aliases = @('account')
            trainable = $false
            benchmark = 'data\processed\benchmark_account_tool_boundary_native_sft.jsonl'
        }
        medical = @{
            aliases = @('medical')
            trainable = $false
            benchmark = 'data\processed\benchmark_medical_refusal_boundary_native_sft.jsonl'
        }
        oos_tool = @{
            aliases = @('oos_tool', 'oos-tool', 'oostool')
            trainable = $false
            benchmark = 'data\processed\benchmark_oos_vs_tool_boundary_native_sft.jsonl'
        }
    }
}

function Get-AiLanguageCoreDefaultSettingsPath {
    param(
        [string]$RepoRoot
    )

    return Join-Path $RepoRoot 'config\command_defaults.json'
}

function Get-AiLanguageCoreCanonicalType {
    param(
        [string]$TypeName
    )

    if (-not $TypeName) {
        return $null
    }

    $normalized = $TypeName.Trim().ToLowerInvariant()
    foreach ($entry in (Get-AiLanguageCoreCommandTypeCatalog).GetEnumerator()) {
        if ($entry.Key -eq $normalized -or $entry.Value.aliases -contains $normalized) {
            return $entry.Key
        }
    }

    return $null
}

function Get-AiLanguageCoreSupportedTypeList {
    return @('default', 'core', 'base', 'stress', 'account', 'medical', 'oos_tool')
}

function Get-AiLanguageCoreTrainableTypeList {
    return @('default', 'core', 'base', 'stress')
}

function Resolve-AiLanguageCoreType {
    param(
        [string]$TypeName,
        [switch]$RequireTrainable
    )

    $canonical = Get-AiLanguageCoreCanonicalType -TypeName $TypeName
    if (-not $canonical) {
        $valid = if ($RequireTrainable) {
            (Get-AiLanguageCoreTrainableTypeList) -join ', '
        } else {
            (Get-AiLanguageCoreSupportedTypeList) -join ', '
        }
        throw "Unknown type '$TypeName'. Valid types: $valid."
    }

    $entry = (Get-AiLanguageCoreCommandTypeCatalog)[$canonical]
    if ($RequireTrainable -and -not $entry.trainable) {
        $valid = (Get-AiLanguageCoreTrainableTypeList) -join ', '
        throw "Type '$TypeName' is benchmark-only. Trainable types: $valid."
    }

    return $canonical
}

function Get-AiLanguageCoreDefaultType {
    param(
        [string]$RepoRoot
    )

    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    if (Test-Path $settingsPath) {
        try {
            $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
            $canonical = Resolve-AiLanguageCoreType -TypeName $settings.default_type -RequireTrainable
            if ($canonical) {
                return $canonical
            }
        } catch {
        }
    }

    return 'stress'
}

function Set-AiLanguageCoreDefaultType {
    param(
        [string]$RepoRoot,
        [string]$TypeName
    )

    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable
    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    $settingsDir = Split-Path -Parent $settingsPath
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null
    }

    $payload = [ordered]@{
        default_type = $canonical
        updated_at = (Get-Date).ToString('o')
    }
    $payload | ConvertTo-Json | Set-Content -Path $settingsPath
    return $canonical
}

function Resolve-AiLanguageCoreTrainingConfig {
    param(
        [string]$RepoRoot,
        [string]$TypeName
    )

    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable
    return Join-Path $RepoRoot ((Get-AiLanguageCoreCommandTypeCatalog)[$canonical].config)
}

function Resolve-AiLanguageCoreBenchmarkFile {
    param(
        [string]$RepoRoot,
        [string]$TypeName
    )

    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName
    return Join-Path $RepoRoot ((Get-AiLanguageCoreCommandTypeCatalog)[$canonical].benchmark)
}

function Get-AiLanguageCoreRunType {
    param(
        [string]$RunDir
    )

    $trainingConfigPath = Join-Path $RunDir 'training_config.json'
    if (Test-Path $trainingConfigPath) {
        try {
            $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
            if ($trainingConfig.benchmark_file -and [string]$trainingConfig.benchmark_file -match 'benchmark_stress') {
                return 'stress'
            }
        } catch {
        }
    }

    $runName = Split-Path -Leaf $RunDir
    if ($runName -match '-stress-') {
        return 'stress'
    }

    return 'default'
}

function Get-AiLanguageCoreLatestCompletedRunPath {
    param(
        [string]$RepoRoot,
        [string]$TypeName
    )

    $canonical = if ($TypeName) {
        Resolve-AiLanguageCoreType -TypeName $TypeName
    } else {
        $null
    }

    $statusPath = Get-ChildItem -Path (Join-Path $RepoRoot 'models\runs') -Filter training_status.json -Recurse -File |
        Sort-Object LastWriteTime -Descending |
        Where-Object {
            try {
                $status = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $runDir = Split-Path -Parent $_.FullName
                $runType = Get-AiLanguageCoreRunType -RunDir $runDir
                $typeMatches = (-not $canonical) -or ($runType -eq $canonical)
                $typeMatches -and (Test-Path (Join-Path $runDir 'model.pt')) -and ($status.global_step -gt 0)
            } catch {
                $false
            }
        } |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $statusPath) {
        if ($canonical) {
            throw "Could not find a completed training run for type '$canonical' under $(Join-Path $RepoRoot 'models\runs')"
        }
        throw "Could not find a completed training run under $(Join-Path $RepoRoot 'models\runs')"
    }

    return Split-Path -Parent $statusPath
}
