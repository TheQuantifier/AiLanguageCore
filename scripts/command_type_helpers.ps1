function Get-AiLanguageCoreCommandTypeCatalog {
    return @{
        core = @{
            aliases = @('core')
            optimization_frozen = $false
            categories = @{
                category_prediction = @{
                    trainable = $true
                    config = 'models\configs\v1_native_byte_transformer_category_prediction_config.json'
                    benchmark = 'data\processed\benchmark_category_prediction_sft.jsonl'
                }
                full_response = @{
                    trainable = $true
                    config = 'models\configs\v1_native_byte_transformer_config.json'
                    benchmark = 'data\processed\benchmark_full_response_sft.jsonl'
                }
            }
        }
        stress = @{
            aliases = @('stress')
            optimization_frozen = $false
            categories = @{
                category_prediction = @{
                    trainable = $true
                    config = 'models\configs\v1_native_byte_transformer_stress_config.json'
                    benchmark = 'data\processed\benchmark_stress_native_sft.jsonl'
                }
            }
        }
        stress_v2 = @{
            aliases = @('stress_v2', 'stress-v2', 'stress2')
            optimization_frozen = $true
            categories = @{
                category_prediction = @{
                    trainable = $true
                    config = 'models\configs\v1_native_byte_transformer_stress_v2_config.json'
                    benchmark = 'data\processed\benchmark_stress_v2_native_sft.jsonl'
                }
            }
        }
        account = @{
            aliases = @('account')
            optimization_frozen = $true
            categories = @{
                category_prediction = @{
                    trainable = $false
                    benchmark = 'data\processed\benchmark_account_tool_boundary_native_sft.jsonl'
                }
            }
        }
        medical = @{
            aliases = @('medical')
            optimization_frozen = $true
            categories = @{
                category_prediction = @{
                    trainable = $false
                    benchmark = 'data\processed\benchmark_medical_refusal_boundary_native_sft.jsonl'
                }
            }
        }
        oos_tool = @{
            aliases = @('oos_tool', 'oos-tool', 'oostool')
            optimization_frozen = $true
            categories = @{
                category_prediction = @{
                    trainable = $false
                    benchmark = 'data\processed\benchmark_oos_vs_tool_boundary_native_sft.jsonl'
                }
            }
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

function Get-AiLanguageCoreCanonicalCategory {
    param(
        [string]$CategoryName
    )

    if (-not $CategoryName) {
        return $null
    }

    $normalized = $CategoryName.Trim().ToLowerInvariant()
    switch ($normalized) {
        'category_prediction' { return 'category_prediction' }
        'category-prediction' { return 'category_prediction' }
        'prediction' { return 'category_prediction' }
        'full_response' { return 'full_response' }
        'full-response' { return 'full_response' }
        'response' { return 'full_response' }
        default { return $null }
    }
}

function Get-AiLanguageCoreSupportedTypeList {
    return @('default', 'core', 'stress', 'stress_v2', 'account', 'medical', 'oos_tool')
}

function Get-AiLanguageCoreSupportedCategoryList {
    return @('default', 'category_prediction', 'full_response')
}

function Get-AiLanguageCoreTrainableTypeList {
    return @('default', 'core', 'stress', 'stress_v2')
}

function Get-AiLanguageCoreDefaultCommandList {
    return @('train', 'benchmark', 'frozen_benchmark', 'autotrain', 'improve')
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

    if ($RequireTrainable) {
        $catalogEntry = (Get-AiLanguageCoreCommandTypeCatalog)[$canonical]
        $hasTrainableCategory = $false
        foreach ($categoryEntry in $catalogEntry.categories.Values) {
            if ($categoryEntry.trainable) {
                $hasTrainableCategory = $true
                break
            }
        }
        if (-not $hasTrainableCategory) {
            $valid = (Get-AiLanguageCoreTrainableTypeList) -join ', '
            throw "Type '$TypeName' is benchmark-only. Trainable types: $valid."
        }
    }

    return $canonical
}

function Resolve-AiLanguageCoreCategory {
    param(
        [string]$CategoryName
    )

    $canonical = Get-AiLanguageCoreCanonicalCategory -CategoryName $CategoryName
    if (-not $canonical) {
        throw "Unknown category '$CategoryName'. Valid categories: $((Get-AiLanguageCoreSupportedCategoryList) -join ', ')."
    }
    return $canonical
}

function Get-AiLanguageCorePreferredCategoryForType {
    param(
        [string]$TypeName
    )

    $canonicalType = Resolve-AiLanguageCoreType -TypeName $TypeName
    if ($canonicalType -eq 'core') {
        return 'full_response'
    }
    return 'category_prediction'
}

function Get-AiLanguageCoreCategoryEntry {
    param(
        [string]$TypeName,
        [string]$CategoryName,
        [switch]$RequireTrainable
    )

    $canonicalType = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$RequireTrainable
    $canonicalCategory = Resolve-AiLanguageCoreCategory -CategoryName $CategoryName
    $typeEntry = (Get-AiLanguageCoreCommandTypeCatalog)[$canonicalType]
    $categoryEntry = $typeEntry.categories[$canonicalCategory]
    if (-not $categoryEntry) {
        throw "Category '$canonicalCategory' is not supported for type '$canonicalType'."
    }
    if ($RequireTrainable -and -not $categoryEntry.trainable) {
        throw "Type '$canonicalType' with category '$canonicalCategory' is benchmark-only and cannot be trained."
    }
    return $categoryEntry
}

function Get-AiLanguageCoreStoredDefaults {
    param(
        [string]$RepoRoot
    )

    $fallback = [ordered]@{
        global = [ordered]@{
            type = 'core'
            category = 'full_response'
        }
        command_defaults = [ordered]@{
            train = [ordered]@{ type = 'core'; category = 'full_response' }
            benchmark = [ordered]@{ type = 'core'; category = 'full_response' }
            frozen_benchmark = [ordered]@{ type = 'stress_v2'; category = 'category_prediction' }
            autotrain = [ordered]@{ type = 'core'; category = 'full_response' }
            improve = [ordered]@{ type = 'core'; category = 'full_response' }
        }
    }

    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    if (-not (Test-Path $settingsPath)) {
        return $fallback
    }

    try {
        $raw = Get-Content $settingsPath -Raw | ConvertFrom-Json
    } catch {
        return $fallback
    }

    $globalType = if ($raw.global_defaults.type) { [string]$raw.global_defaults.type } elseif ($raw.default_type) { [string]$raw.default_type } else { $fallback.global.type }
    if ($globalType -eq 'base') { $globalType = 'core' }
    $globalType = Resolve-AiLanguageCoreType -TypeName $globalType -RequireTrainable

    $globalCategory = if ($raw.global_defaults.category) { [string]$raw.global_defaults.category } elseif ($raw.default_category) { [string]$raw.default_category } else { $null }
    if ($globalCategory) {
        $globalCategory = Resolve-AiLanguageCoreCategory -CategoryName $globalCategory
    } else {
        $globalCategory = Get-AiLanguageCorePreferredCategoryForType -TypeName $globalType
    }
    $null = Get-AiLanguageCoreCategoryEntry -TypeName $globalType -CategoryName $globalCategory -RequireTrainable

    $commandDefaults = [ordered]@{}
    foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
        $saved = if ($raw.command_defaults) { $raw.command_defaults.$name } else { $null }
        if ($saved -is [string]) {
            $savedType = [string]$saved
            if ($savedType -eq 'base') { $savedType = 'core' }
            $savedType = Resolve-AiLanguageCoreType -TypeName $savedType -RequireTrainable:(($name -ne 'benchmark') -and ($name -ne 'frozen_benchmark'))
            $savedCategory = Get-AiLanguageCorePreferredCategoryForType -TypeName $savedType
        } elseif ($saved) {
            $savedType = if ($saved.type) { [string]$saved.type } else { $globalType }
            if ($savedType -eq 'base') { $savedType = 'core' }
            $savedType = Resolve-AiLanguageCoreType -TypeName $savedType -RequireTrainable:(($name -ne 'benchmark') -and ($name -ne 'frozen_benchmark'))
            $savedCategory = if ($saved.category) { Resolve-AiLanguageCoreCategory -CategoryName ([string]$saved.category) } else { Get-AiLanguageCorePreferredCategoryForType -TypeName $savedType }
        } else {
            $savedType = $fallback.command_defaults[$name].type
            $savedCategory = $fallback.command_defaults[$name].category
        }

        try {
            $null = Get-AiLanguageCoreCategoryEntry -TypeName $savedType -CategoryName $savedCategory -RequireTrainable:(($name -ne 'benchmark') -and ($name -ne 'frozen_benchmark'))
        } catch {
            $savedCategory = Get-AiLanguageCorePreferredCategoryForType -TypeName $savedType
            $null = Get-AiLanguageCoreCategoryEntry -TypeName $savedType -CategoryName $savedCategory -RequireTrainable:(($name -ne 'benchmark') -and ($name -ne 'frozen_benchmark'))
        }

        $commandDefaults[$name] = [ordered]@{
            type = $savedType
            category = $savedCategory
        }
    }

    return [ordered]@{
        global = [ordered]@{
            type = $globalType
            category = $globalCategory
        }
        command_defaults = $commandDefaults
    }
}

function Get-AiLanguageCoreDefaultSelection {
    param(
        [string]$RepoRoot,
        [string]$CommandName
    )

    $defaults = Get-AiLanguageCoreStoredDefaults -RepoRoot $RepoRoot
    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    if ($resolvedCommand) {
        $commandSelection = $defaults.command_defaults[$resolvedCommand]
        if ($commandSelection) {
            return [pscustomobject]@{
                Type = [string]$commandSelection.type
                Category = [string]$commandSelection.category
            }
        }
    }

    return [pscustomobject]@{
        Type = [string]$defaults.global.type
        Category = [string]$defaults.global.category
    }
}

function Get-AiLanguageCoreDefaultType {
    param(
        [string]$RepoRoot,
        [string]$CommandName
    )

    return (Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName $CommandName).Type
}

function Get-AiLanguageCoreDefaultCategory {
    param(
        [string]$RepoRoot,
        [string]$CommandName
    )

    return (Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName $CommandName).Category
}

function Resolve-AiLanguageCoreSelection {
    param(
        [string]$RepoRoot,
        [string]$CommandName,
        [string]$TypeName,
        [string]$CategoryName,
        [switch]$RequireTrainable
    )

    $defaultSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName $CommandName
    $effectiveType = if ($TypeName -and $TypeName.Trim().ToLowerInvariant() -ne 'default') {
        Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$RequireTrainable
    } else {
        $defaultSelection.Type
    }

    $effectiveCategory = if ($CategoryName -and $CategoryName.Trim().ToLowerInvariant() -ne 'default') {
        Resolve-AiLanguageCoreCategory -CategoryName $CategoryName
    } elseif ($TypeName -and $TypeName.Trim().ToLowerInvariant() -ne 'default' -and -not $CategoryName) {
        Get-AiLanguageCorePreferredCategoryForType -TypeName $effectiveType
    } else {
        $defaultSelection.Category
    }

    $null = Get-AiLanguageCoreCategoryEntry -TypeName $effectiveType -CategoryName $effectiveCategory -RequireTrainable:$RequireTrainable
    return [pscustomobject]@{
        Type = $effectiveType
        Category = $effectiveCategory
    }
}

function Set-AiLanguageCoreDefaultSelection {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CategoryName,
        [string]$CommandName
    )

    $normalizedType = if ($TypeName) { $TypeName.Trim().ToLowerInvariant() } else { '' }
    if ($normalizedType -eq 'default') {
        throw "Cannot set a command default to the symbolic type 'default'. Use a concrete type."
    }

    $resolvedCommand = Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName
    $requireTrainable = -not ($resolvedCommand -in @('benchmark', 'frozen_benchmark'))
    $canonicalType = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$requireTrainable
    $canonicalCategory = if ($CategoryName) {
        Resolve-AiLanguageCoreCategory -CategoryName $CategoryName
    } else {
        Get-AiLanguageCorePreferredCategoryForType -TypeName $canonicalType
    }
    $null = Get-AiLanguageCoreCategoryEntry -TypeName $canonicalType -CategoryName $canonicalCategory -RequireTrainable:$requireTrainable

    if ($resolvedCommand -and $resolvedCommand -notin @('benchmark', 'frozen_benchmark') -and (Test-AiLanguageCoreTypeOptimizationFrozen -TypeName $canonicalType)) {
        throw "Cannot save '$canonicalType' as the default for '$resolvedCommand' because it is frozen as a regression benchmark."
    }
    if (-not $resolvedCommand -and (Test-AiLanguageCoreTypeOptimizationFrozen -TypeName $canonicalType)) {
        throw "Cannot save '$canonicalType' as the global default because it is frozen as a regression benchmark."
    }

    $settingsPath = Get-AiLanguageCoreDefaultSettingsPath -RepoRoot $RepoRoot
    $settingsDir = Split-Path -Parent $settingsPath
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null
    }

    $defaults = Get-AiLanguageCoreStoredDefaults -RepoRoot $RepoRoot
    if ($resolvedCommand) {
        $defaults.command_defaults[$resolvedCommand] = [ordered]@{
            type = $canonicalType
            category = $canonicalCategory
        }
    } else {
        $defaults.global = [ordered]@{
            type = $canonicalType
            category = $canonicalCategory
        }
        foreach ($name in (Get-AiLanguageCoreDefaultCommandList)) {
            $defaults.command_defaults[$name] = [ordered]@{
                type = $canonicalType
                category = $canonicalCategory
            }
        }
    }

    $payload = [ordered]@{
        global_defaults = [ordered]@{
            type = $defaults.global.type
            category = $defaults.global.category
        }
        command_defaults = $defaults.command_defaults
        updated_at = (Get-Date).ToString('o')
    }
    $json = [string]($payload | ConvertTo-Json -Depth 6)
    [System.IO.File]::WriteAllText($settingsPath, $json)

    return [pscustomobject]@{
        Type = $canonicalType
        Category = $canonicalCategory
    }
}

function Set-AiLanguageCoreDefaultTypeOnly {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CommandName
    )

    $resolvedCommand = if ($CommandName) { Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName } else { $null }
    $currentSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName $resolvedCommand
    $requireTrainable = -not ($resolvedCommand -in @('benchmark', 'frozen_benchmark'))
    $canonicalType = Resolve-AiLanguageCoreType -TypeName $TypeName -RequireTrainable:$requireTrainable
    $targetCategory = $currentSelection.Category

    try {
        $null = Get-AiLanguageCoreCategoryEntry -TypeName $canonicalType -CategoryName $targetCategory -RequireTrainable:$requireTrainable
    } catch {
        $targetCategory = Get-AiLanguageCorePreferredCategoryForType -TypeName $canonicalType
    }

    return Set-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -TypeName $canonicalType -CategoryName $targetCategory -CommandName $resolvedCommand
}

function Set-AiLanguageCoreDefaultCategoryOnly {
    param(
        [string]$RepoRoot,
        [string]$CategoryName,
        [string]$CommandName
    )

    $resolvedCommand = if ($CommandName) { Resolve-AiLanguageCoreDefaultCommandName -CommandName $CommandName } else { $null }
    $currentSelection = Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName $resolvedCommand
    $canonicalCategory = Resolve-AiLanguageCoreCategory -CategoryName $CategoryName
    $requireTrainable = -not ($resolvedCommand -in @('benchmark', 'frozen_benchmark'))
    $null = Get-AiLanguageCoreCategoryEntry -TypeName $currentSelection.Type -CategoryName $canonicalCategory -RequireTrainable:$requireTrainable

    return Set-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -TypeName $currentSelection.Type -CategoryName $canonicalCategory -CommandName $resolvedCommand
}

function Resolve-AiLanguageCoreTrainingConfig {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CategoryName
    )

    $entry = Get-AiLanguageCoreCategoryEntry -TypeName $TypeName -CategoryName $CategoryName -RequireTrainable
    if (-not $entry.config) {
        throw "No training config defined for type '$TypeName' category '$CategoryName'."
    }
    return Join-Path $RepoRoot $entry.config
}

function Test-AiLanguageCoreTypeOptimizationFrozen {
    param(
        [string]$TypeName
    )

    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName
    $entry = (Get-AiLanguageCoreCommandTypeCatalog)[$canonical]
    return [bool]$entry.optimization_frozen
}

function Assert-AiLanguageCoreTypeOptimizable {
    param(
        [string]$TypeName,
        [string]$CommandName
    )

    $canonical = Resolve-AiLanguageCoreType -TypeName $TypeName
    if (Test-AiLanguageCoreTypeOptimizationFrozen -TypeName $canonical) {
        $commandText = if ($CommandName) { $CommandName } else { 'This command' }
        throw "$commandText cannot target '$canonical' because it is frozen as a regression benchmark. Use 'benchmark $canonical category_prediction' to verify it, and switch train/autotrain/improve to an active type."
    }
}

function Resolve-AiLanguageCoreBenchmarkFile {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CategoryName
    )

    $entry = Get-AiLanguageCoreCategoryEntry -TypeName $TypeName -CategoryName $CategoryName
    return Join-Path $RepoRoot $entry.benchmark
}

function Get-AiLanguageCoreFrozenCandidateSelections {
    param(
        [string]$CategoryName
    )

    $canonicalCategory = Resolve-AiLanguageCoreCategory -CategoryName $CategoryName
    $results = @()
    foreach ($typeName in @('stress_v2', 'account', 'medical', 'oos_tool', 'stress', 'core')) {
        $typeEntry = (Get-AiLanguageCoreCommandTypeCatalog)[$typeName]
        if (-not $typeEntry) {
            continue
        }
        if (-not [bool]$typeEntry.optimization_frozen) {
            continue
        }
        if ($typeEntry.categories[$canonicalCategory]) {
            $results += [pscustomobject]@{
                Type = $typeName
                Category = $canonicalCategory
            }
        }
    }
    return $results
}

function Resolve-AiLanguageCoreFrozenBenchmarkSelection {
    param(
        [string]$RepoRoot,
        [string]$CurrentTypeName,
        [string]$CurrentCategoryName
    )

    $currentType = Resolve-AiLanguageCoreType -TypeName $CurrentTypeName
    $currentCategory = Resolve-AiLanguageCoreCategory -CategoryName $CurrentCategoryName

    if ((Test-AiLanguageCoreTypeOptimizationFrozen -TypeName $currentType)) {
        try {
            $null = Get-AiLanguageCoreCategoryEntry -TypeName $currentType -CategoryName $currentCategory
            return [pscustomobject]@{
                Type = $currentType
                Category = $currentCategory
                Source = 'current_selection'
            }
        } catch {
        }
    }

    $categoryCandidates = Get-AiLanguageCoreFrozenCandidateSelections -CategoryName $currentCategory
    if ($categoryCandidates.Count -gt 0) {
        return [pscustomobject]@{
            Type = $categoryCandidates[0].Type
            Category = $categoryCandidates[0].Category
            Source = 'category_fallback'
        }
    }

    $savedFrozen = Get-AiLanguageCoreDefaultSelection -RepoRoot $RepoRoot -CommandName 'frozen_benchmark'
    return [pscustomobject]@{
        Type = $savedFrozen.Type
        Category = $savedFrozen.Category
        Source = 'saved_frozen_benchmark'
    }
}

function Get-AiLanguageCoreRunType {
    param(
        [string]$RunDir
    )

    $trainingConfigPath = Join-Path $RunDir 'training_config.json'
    if (Test-Path $trainingConfigPath) {
        try {
            $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
            if ($trainingConfig.output_dir -and [string]$trainingConfig.output_dir -match 'stress-v2') { return 'stress_v2' }
            if ($trainingConfig.output_dir -and [string]$trainingConfig.output_dir -match 'stress') { return 'stress' }
        } catch {
        }
    }

    $runName = Split-Path -Leaf $RunDir
    if ($runName -match '-stress-v2-') { return 'stress_v2' }
    if ($runName -match '-stress-') { return 'stress' }
    return 'core'
}

function Get-AiLanguageCoreRunCategory {
    param(
        [string]$RunDir
    )

    $trainingConfigPath = Join-Path $RunDir 'training_config.json'
    if (Test-Path $trainingConfigPath) {
        try {
            $trainingConfig = Get-Content $trainingConfigPath -Raw | ConvertFrom-Json
            if ($trainingConfig.train_file -and [string]$trainingConfig.train_file -match 'full_response') {
                return 'full_response'
            }
            if ($trainingConfig.train_file -and [string]$trainingConfig.train_file -match 'category_prediction') {
                return 'category_prediction'
            }
            if ($trainingConfig.benchmark_file -and [string]$trainingConfig.benchmark_file -match 'full_response') {
                return 'full_response'
            }
        } catch {
        }
    }

    $runName = Split-Path -Leaf $RunDir
    if ($runName -match 'full-response') { return 'full_response' }
    if ($runName -match 'category-prediction') { return 'category_prediction' }
    return 'category_prediction'
}

function Get-AiLanguageCoreLatestCompletedRunPath {
    param(
        [string]$RepoRoot,
        [string]$TypeName,
        [string]$CategoryName
    )

    $canonicalType = if ($TypeName) { Resolve-AiLanguageCoreType -TypeName $TypeName } else { $null }
    $canonicalCategory = if ($CategoryName) { Resolve-AiLanguageCoreCategory -CategoryName $CategoryName } else { $null }

    $statusPath = Get-ChildItem -Path (Join-Path $RepoRoot 'models\runs') -Filter training_status.json -Recurse -File |
        Sort-Object LastWriteTime -Descending |
        Where-Object {
            try {
                $status = Get-Content $_.FullName -Raw | ConvertFrom-Json
                $runDir = Split-Path -Parent $_.FullName
                $runType = Get-AiLanguageCoreRunType -RunDir $runDir
                $runCategory = Get-AiLanguageCoreRunCategory -RunDir $runDir
                $typeMatches = (-not $canonicalType) -or ($runType -eq $canonicalType)
                $categoryMatches = (-not $canonicalCategory) -or ($runCategory -eq $canonicalCategory)
                $typeMatches -and $categoryMatches -and (Test-Path (Join-Path $runDir 'model.pt')) -and ($status.global_step -gt 0)
            } catch {
                $false
            }
        } |
        Select-Object -First 1 -ExpandProperty FullName

    if (-not $statusPath) {
        if ($canonicalType -and $canonicalCategory) {
            throw "Could not find a completed training run for type '$canonicalType' category '$canonicalCategory' under $(Join-Path $RepoRoot 'models\runs')"
        }
        throw "Could not find a completed training run under $(Join-Path $RepoRoot 'models\runs')"
    }

    return Split-Path -Parent $statusPath
}
