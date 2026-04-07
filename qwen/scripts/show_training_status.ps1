param(
    [string]$RunDir = "qwen/models/runs/v1-qwen2.5-0.5b-lora-cpu",
    [switch]$Watch,
    [int]$IntervalSeconds = 5
)

function Resolve-RunDir {
    param([string]$RequestedRunDir)

    if (Test-Path $RequestedRunDir) {
        return $RequestedRunDir
    }

    $parent = Split-Path -Parent $RequestedRunDir
    $leaf = Split-Path -Leaf $RequestedRunDir
    if (-not $parent) {
        $parent = "."
    }
    if (-not (Test-Path $parent)) {
        return $RequestedRunDir
    }

    $matches = Get-ChildItem -Path $parent -Directory |
        Where-Object { $_.Name -like "$leaf-*" } |
        Sort-Object LastWriteTimeUtc -Descending

    if ($matches) {
        return $matches[0].FullName
    }

    return $RequestedRunDir
}

$resolvedRunDir = Resolve-RunDir -RequestedRunDir $RunDir
$statusPath = Join-Path $resolvedRunDir "training_status.json"

function Show-Status {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        Write-Host "No training status file found at $Path"
        return
    }

    try {
        $status = Get-Content -Raw -Path $Path | ConvertFrom-Json
    }
    catch {
        Write-Host "Could not parse $Path"
        return
    }

    Clear-Host
    Write-Host "Training Status"
    Write-Host "Path: $Path"
    Write-Host ""
    Write-Host ("Status:           {0}" -f $status.status)
    Write-Host ("Base model:       {0}" -f $status.base_model)
    Write-Host ("PID:              {0}" -f $status.pid)
    Write-Host ("Device:           {0}" -f $status.device)
    Write-Host ("Started at:       {0}" -f $status.started_at)
    Write-Host ("Updated at:       {0}" -f $status.updated_at)
    Write-Host ("Completed at:     {0}" -f $status.completed_at)
    Write-Host ("Global step:      {0}" -f $status.global_step)
    Write-Host ("Max steps:        {0}" -f $status.max_steps)
    Write-Host ("Epoch:            {0}" -f $status.epoch)
    Write-Host ("Train examples:   {0}" -f $status.train_examples)
    Write-Host ("Validation ex:    {0}" -f $status.validation_examples)
    Write-Host ("Last checkpoint:  {0}" -f $status.last_checkpoint)
    Write-Host ""
    Write-Host "Latest log:"

    if ($status.latest_log) {
        $status.latest_log.PSObject.Properties |
            Sort-Object Name |
            ForEach-Object {
                Write-Host ("  {0}: {1}" -f $_.Name, $_.Value)
            }
    }
    else {
        Write-Host "  <none yet>"
    }

    if ($status.error) {
        Write-Host ""
        Write-Host ("Error: {0}" -f $status.error)
    }

    if ($status.traceback) {
        Write-Host ""
        Write-Host "Traceback:"
        Write-Host $status.traceback
    }
}

if ($Watch) {
    while ($true) {
        Show-Status -Path $statusPath
        if (Test-Path $statusPath) {
            $status = Get-Content -Raw -Path $statusPath | ConvertFrom-Json
            if ($status.status -in @("completed", "failed")) {
                break
            }
        }
        Start-Sleep -Seconds $IntervalSeconds
    }
}
else {
    Show-Status -Path $statusPath
}
