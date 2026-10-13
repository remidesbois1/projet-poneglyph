param(
    [switch]$Build,
    [switch]$Export,
    [switch]$Smoke,
    [switch]$Recalibrate,
    [string]$RunName = 'rtx5090-20261011'
)
$ErrorActionPreference = 'Stop'
$taskDirectory = $PSScriptRoot
$taskRepository = [IO.Path]::GetFullPath((Join-Path $taskDirectory '..\..'))
$taskImage = 'poneglyph/lighton3-bbox:5090'
$activeTraining = @(docker ps --filter 'name=lighton3-bbox-training' --format '{{.Names}}') + @(docker ps --filter 'name=lighton3-bbox-smoke' --format '{{.Names}}')
$activeTraining += @(docker ps --format '{{.Names}}' | Where-Object { $_ -match '^lighton3-cer-refinement(-smoke)?$' })
if ($activeTraining.Count -gt 0) { throw 'A LightOnOCR-3 run is active. Use its logs/dashboard before launching or exporting again.' }
if ($Build) {
    docker build -t $taskImage -f (Join-Path $taskDirectory 'Dockerfile') $taskRepository
    if ($LASTEXITCODE -ne 0) { throw 'Image build failed' }
}
$mount = "${taskDirectory}:/workspace"
if ($Export) {
    $exportArgs = @('run', '--rm', '--name', 'lighton3-export')
    $backendEnv = Join-Path $taskRepository 'backend\.env'
    if (Test-Path -LiteralPath $backendEnv) { $exportArgs += @('--env-file', $backendEnv) }
    $exportArgs += @('--env-file', (Join-Path $taskRepository '.env'), '-v', $mount, $taskImage, 'export_dataset.py')
    & docker @exportArgs
    if ($LASTEXITCODE -ne 0) { throw 'Dataset export failed' }
}
if (-not (Test-Path -LiteralPath (Join-Path $taskDirectory 'dataset\dataset_report.json'))) { throw 'Export the dataset first' }
$container = if ($Smoke) { 'lighton3-bbox-smoke' } else { 'lighton3-bbox-training' }
$runArgs = @('run', '-d', '--name', $container, '--gpus', 'all', '--shm-size', '2g', '-v', $mount, $taskImage, 'train.py', '--dataset', '/workspace/dataset', '--output', "/workspace/outputs/$RunName")
if ($Smoke) { $runArgs += @('--smoke-steps', '2') }
if ($Recalibrate) { $runArgs += '--recalibrate' }
& docker @runArgs
if ($LASTEXITCODE -ne 0) { throw 'Training launch failed; inspect the existing container before relaunching' }
Write-Host "Suivi : docker logs -f $container"
Write-Host "Dashboard : $taskDirectory\outputs\$RunName\training_dashboard.html"
