param(
    [switch]$Build,
    [switch]$Smoke,
    [string]$RunName = 'rtx5090-20261011-cer-refinement',
    [string]$InitialCheckpoint = 'rtx5090-20261011/checkpoint-333'
)
$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$taskRepository = [IO.Path]::GetFullPath((Join-Path $taskDirectory '..\..'))
$taskImage = 'poneglyph/lighton3-bbox:5090'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$' -or $InitialCheckpoint -notmatch '^[a-zA-Z0-9_/-]+$' -or $InitialCheckpoint.Contains('..')) { throw 'Invalid local run/checkpoint path' }
$taskActive = @(docker ps --format '{{.Names}}' | Where-Object { $_ -match '^lighton3-(bbox-training|bbox-smoke|cer-refinement|cer-refinement-smoke)$' })
if ($taskActive.Count -gt 0) { throw 'An OCR GPU job is active; inspect its progress before launching another.' }
$taskInitialPath = Join-Path $taskDirectory "outputs\$InitialCheckpoint"
if (-not (Test-Path -LiteralPath (Join-Path $taskInitialPath 'config.json'))) { throw 'Starting native checkpoint is missing.' }
if ($Build) {
    docker build -t $taskImage -f (Join-Path $taskDirectory 'Dockerfile') $taskRepository
    if ($LASTEXITCODE -ne 0) { throw 'Image build failed' }
}
$taskContainer = if ($Smoke) { 'lighton3-cer-refinement-smoke' } else { 'lighton3-cer-refinement' }
$taskArgs = @('run', '-d', '--name', $taskContainer, '--gpus', 'all', '--shm-size', '2g', '-v', "${taskDirectory}:/workspace", $taskImage,
    'train.py', '--dataset', '/workspace/dataset', '--output', "/workspace/outputs/$RunName",
    '--init-model', "/workspace/outputs/$InitialCheckpoint", '--epochs', '3', '--lr', '3e-6',
    '--selection-metric', 'cer', '--bbox-loss-weight', '0.2', '--eos-loss-weight', '2',
    '--no-augmentation', '--generation-pages', '0', '--max-physical-batch', '1')
if ($Smoke) { $taskArgs += @('--smoke-steps', '2', '--skip-baseline', '--batch', '1') }
& docker @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Training launch failed; inspect the named container.' }
Write-Host "Suivi : docker logs -f $taskContainer"
Write-Host "Dashboard : outputs\$RunName\training_dashboard.html"
