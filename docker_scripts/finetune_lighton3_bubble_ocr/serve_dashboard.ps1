param([int]$Port = 8768, [string]$RunName = 'rtx5090-20261012-bubble')
$ErrorActionPreference = 'Stop'
if ($RunName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid run name' }
$taskDirectory = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "outputs\$RunName"))
if (-not (Test-Path -LiteralPath $taskDirectory)) { throw 'Launch the run first' }
$taskContainer = "lighton3-bubble-dashboard-$RunName"
$taskExisting = @(docker ps -a --filter "name=^/$taskContainer`$" --format '{{.Names}}')
if ($taskExisting.Count -gt 0) { docker start $taskContainer }
else {
    docker run -d --name $taskContainer -p "127.0.0.1:${Port}:8766" -v "${taskDirectory}:/dashboard:ro" --entrypoint python poneglyph/lighton3-bubble:5090 -m http.server 8766 --bind 0.0.0.0 --directory /dashboard
}
if ($LASTEXITCODE -ne 0) { throw 'Dashboard launch failed' }
Write-Host "http://127.0.0.1:$Port/training_dashboard.html"
