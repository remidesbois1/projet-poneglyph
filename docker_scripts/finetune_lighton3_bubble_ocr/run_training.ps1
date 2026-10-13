param(
    [ValidateSet('train', 'baseline', 'calibrate', 'test', 'finalize', 'verify-native')][string]$Mode = 'train',
    [int]$Batch = 16,
    [int]$Accum = 2,
    [int]$SmokeSteps = 0,
    [string]$RunName = 'rtx5090-20261012-bubble',
    [switch]$Build,
    [switch]$DeferTest,
    [string]$InitModel = '',
    [string]$LearningRate = '1.2e-5',
    [string]$Epochs = '6',
    [int]$WarmupSteps = 60,
    [switch]$NoAugment,
    [double]$EosWeight = 1.0
)
$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$taskImage = 'poneglyph/lighton3-bubble:5090'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run name' }
if ($Batch -le 0 -or $Accum -le 0) { throw 'Batch and accumulation must be positive' }
$active = @(docker ps --format '{{.Names}}' | Where-Object { $_ -match '^lighton3-bubble-(train|calibrate|baseline|test|finalize|verify-native)' })
if ($active.Count -gt 0) { throw "A bubble workload is active: $($active -join ', ')" }
if ($Build) {
    docker build -t $taskImage -f (Join-Path $PSScriptRoot 'Dockerfile') $taskRoot
    if ($LASTEXITCODE -ne 0) { throw 'Image build failed' }
}
$taskData = Join-Path $taskRoot 'docker_scripts\finetune_surya_bubble_ocr\surya_bubble_dataset'
$taskCache = Join-Path $env:USERPROFILE '.cache\huggingface'
$taskContainer = "lighton3-bubble-$Mode-$RunName"
if ($Mode -eq 'calibrate') { $taskContainer += "-b$Batch" }
$taskArgs = @('run', '-d', '--name', $taskContainer, '--gpus', 'all', '--shm-size', '2g',
    '-v', "${PSScriptRoot}:/workspace", '-v', "${taskData}:/data/bubbles:ro", '-v', "${taskCache}:/hf-cache",
    '-v', 'poneglyph-surya-kernels:/kernel-cache', '-e', "LIGHTON3_BUBBLE_OUTPUT_DIR=/workspace/outputs/$RunName",
    '-e', "LIGHTON3_BUBBLE_TRAIN_BATCH=$Batch", '-e', "LIGHTON3_BUBBLE_GRAD_ACCUM=$Accum",
    '-e', "LIGHTON3_BUBBLE_EVAL_BATCH=$Batch", '-e', "LIGHTON3_BUBBLE_LR=$LearningRate",
    '-e', "LIGHTON3_BUBBLE_EPOCHS=$Epochs", '-e', "LIGHTON3_BUBBLE_WARMUP_STEPS=$WarmupSteps",
    '-e', "LIGHTON3_BUBBLE_EOS_WEIGHT=$($EosWeight.ToString([Globalization.CultureInfo]::InvariantCulture))")
if ($DeferTest) { $taskArgs += @('-e', 'LIGHTON3_BUBBLE_DEFER_TEST=1') }
if ($InitModel) { $taskArgs += @('-e', "LIGHTON3_BUBBLE_INIT_MODEL=$InitModel") }
if ($NoAugment) { $taskArgs += @('-e', 'LIGHTON3_BUBBLE_AUGMENT=0') }
$taskArgs += @($taskImage, 'train.py', '--mode', $Mode)
if ($Mode -eq 'calibrate') { $taskArgs += @('--batch-size', "$Batch") }
if ($SmokeSteps -gt 0) { $taskArgs += @('--smoke-steps', "$SmokeSteps") }
& docker @taskArgs
if ($LASTEXITCODE -ne 0) { throw 'Training launch failed; inspect the named container' }
Write-Host "Suivi : docker logs -f $taskContainer"
Write-Host "Rapport : $PSScriptRoot\outputs\$RunName\training_dashboard.html"
