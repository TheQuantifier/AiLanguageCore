function Get-AiLanguageCoreCommandTypeCatalog {
    return @{
        core = @{
            aliases = @('core')
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
        stress_v2 = @{
            aliases = @('stress_v2', 'stress-v2', 'stress2')
            trainable = $true
            config = 'models\configs\v1_native_byte_transformer_stress_v2_config.json'
            benchmark = 'data\processed\benchmark_stress_v2_native_sft.jsonl'
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
    return @('default', 'core', 'stress', 'stress_v2', 'account', 'medical', 'oos_tool')
}

function Get-AiLanguageCoreTrainableTypeList {
    return @('default', 'core', 'stress', 'stress_v2')
}

function Get-AiLanguageCoreDefaultCommandList {
    return @('train', 'benchmark', 'autotrain', 'improve')
}

function Resolve-AiLanguageCoreDefaultCommandName {
    param(
        [string]$CommandName
    )

    if (-not $CommandName) {
        return $null
    }

    $normalized = $CommandName.Trim().ToLowerInvariant()
    if ((Get-AiLanguageCoreDefaultCommandList) -contains $normalized) {
        return $normalized
    }

    throw "Unknown command '$CommandName'. Valid commands: $((Get-AiLanguageCoreDefaultCommandList) -join ', ')."
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

function Resolve-AiLanguageCoreRequestedType {
    param(
        [string]$RepoRoot,
        [string]$CommandName,
        [string]$TypeName,
        [switch]$RequireTrainable
    )

    if (-not $TypeName) {
        return $null
    }

    $normalized = $TypeName.Trim().ToLowerInvariant()
    if ($normalized -eq 'default') {
        return Get-AiLanguageCoreDefaultType -RepoRoot $RepoRoot -CommandName $CommandName
    }

    return Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$RequireTrainable
}

function Get-AiLanguageCoreDefaultType {
    param(
        [string]$RepoRoot,
        [string]$CommandName
    )

    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    if (Test-Path $settingsPath) {
        try {
            $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
            if ($resolvedCommand -and $settings.command_defaults) {
                $savedCommandType = $settings.command_defaults.$resolvedCommand
                if ($savedCommandType) {
                    if ([string]$savedCommandType -eq 'base') {
                        $savedCommandType = 'core'
                    }
                    $requireTrainable = $resolvedCommand -ne 'benchmark'
                    $canonicalForCommand = Resolve-AiLanguageCoreType -TypeName $savedCommandType -RequireTrainable:$requireTrainable
                    if ($canonicalForCommand) {
                        return $canonicalForCommand
                    }
                }
            }

            if ($settings.default_type) {
                $globalType = [string]$settings.default_type
                if ($globalType -eq 'base') {
                    $globalType = 'core'
                }
                $requireTrainableForGlobal = -not ($resolvedCommand -eq 'benchmark')
                $canonical = Resolve-AiLanguageCoreType -TypeName $globalType -RequireTrainable:$requireTrainableForGlobal
                if ($canonical) {
                    return $canonical
                }
            }
        } catch {
        }
    }

    return 'stress'
}

function Set-AiLanguageCoreDefaultType {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CommandName
    )

    $normalizedType = if ($TypeName) { $TypeName.Trim().ToLowerInvariant() } else { '' }
    if ($normalizedType -eq 'default') {
        throw "Cannot set a command default to the symbolic type 'default'. Use a concrete type: core/stress/stress_v2 (or account/medical/oos_tool for benchmark)."
    }

    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    $requireTrainable = -not ($resolvedCommand -eq 'benchmark')
    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$requireTrainable
    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    $settingsDir = Split-Path -Parent $settingsPath
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null
    }

    $globalDefaultToPersist = 'stress'
    $existingDefaults = [ordered]@{}
    if (Test-Path $settingsPath) {
        try {
            $existing = Get-Content $settingsPath -Raw | ConvertFrom-Json
            if ($existing.default_type) {
                $existingGlobalType = [string]$existing.default_type
                if ($existingGlobalType -eq 'base') {
                    $existingGlobalType = 'core'
                }
                $existingGlobal = Resolve-AiLanguageCoreType -TypeName $existingGlobalType -RequireTrainable
                if ($existingGlobal) {
                    $globalDefaultToPersist = $existingGlobal
                }
            }
            if ($existing.command_defaults) {
                foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
                    if ($existing.command_defaults.$name) {
                        $saved = [string]$existing.command_defaults.$name
                        if ($saved -eq 'base') {
                            $saved = 'core'
                        }
                        $existingDefaults[$name] = $saved
                    }
                }
            }
        } catch {
        }
    }

    if ($resolvedCommand) {
        $existingDefaults[$resolvedCommand] = $canonical
    } else {
        $globalDefaultToPersist = $canonical
        foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
            $existingDefaults[$name] = $canonical
        }
    }

    $payload = [ordered]@{
        default_type = $globalDefaultToPersist
        command_defaults = $existingDefaults
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
            if ($trainingConfig.benchmark_file -and [string]$trainingConfig.benchmark_file -match 'benchmark_stress_v2') {
                return 'stress_v2'
            }
            if ($trainingConfig.benchmark_file -and [string]$trainingConfig.benchmark_file -match 'benchmark_stress') {
                return 'stress'
            }
        } catch {
        }
    }

    $runName = Split-Path -Leaf $RunDir
    if ($runName -match '-stress-v2-') {
        return 'stress_v2'
    }
    if ($runName -match '-stress-') {
        return 'stress'
    }

    return 'core'
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
