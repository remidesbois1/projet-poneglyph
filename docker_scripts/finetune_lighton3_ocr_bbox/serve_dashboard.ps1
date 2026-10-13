param(
    [int]$Port = 8766,
    [string]$RunName = 'rtx5090-20261011'
)
$ErrorActionPreference = 'Stop'
$dashboardDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "outputs\$RunName"))
if (-not $dashboardDirectory.StartsWith([IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'outputs')) + '\')) { throw 'Invalid run directory' }
if (-not (Test-Path -LiteralPath $dashboardDirectory)) { throw 'Launch the run first' }
$dashboardContainer = "lighton3-dashboard-$RunName"
$dashboardRunning = @(docker ps --filter "name=^/$dashboardContainer`$" --format '{{.Names}}')
if ($dashboardRunning.Count -gt 0) {
    Write-Host "http://127.0.0.1:$Port/training_dashboard.html"
    return
}
$dashboardExisting = @(docker ps -a --filter "name=^/$dashboardContainer`$" --format '{{.Names}}')
if ($dashboardExisting.Count -gt 0) {
    docker start $dashboardContainer
} else {
    docker run -d --name $dashboardContainer -p "127.0.0.1:${Port}:8766" -v "${dashboardDirectory}:/dashboard:ro" --entrypoint python poneglyph/lighton3-bbox:5090 -m http.server 8766 --bind 0.0.0.0 --directory /dashboard
}
if ($LASTEXITCODE -ne 0) { throw 'Dashboard launch failed; check the selected port' }
Write-Host "http://127.0.0.1:$Port/training_dashboard.html"
