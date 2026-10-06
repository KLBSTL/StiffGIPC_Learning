$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$reportPath = Join-Path $taskRoot 'reports/CLEANUP_LOCAL_FACTORS_20261004.json'
if (Test-Path -LiteralPath $reportPath) { throw 'Cleanup report already exists; inspect it before rerunning.' }
$beforeFree = (Get-PSDrive -Name E).Free
$entries = @(foreach ($folder in @('replay_v42_20261003', 'replay_v42_strict_20261003', 'autodl_perf_v43_20261003')) {
    Get-ChildItem -LiteralPath (Join-Path $taskRoot "downloads/$folder") -Recurse -File -Filter factors.bin | ForEach-Object {
        $resolved = (Resolve-Path -LiteralPath $_.FullName).Path
        if (-not $resolved.StartsWith($taskRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or $_.LinkType) {
            throw "Unexpected path or existing link: $resolved"
        }
        [pscustomobject]@{ Path=$resolved; Bytes=$_.Length; SHA256=(Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash }
    }
})
if ($entries.Count -ne 21) { throw "Expected 21 immutable factor outputs, found $($entries.Count)" }
$plan = @(foreach ($group in ($entries | Group-Object SHA256)) {
    if ($group.Count -lt 2) { continue }
    $original = $group.Group[0]
    foreach ($duplicate in $group.Group[1..($group.Count-1)]) {
        [pscustomobject]@{ Path=$duplicate.Path; Original=$original.Path; Bytes=$duplicate.Bytes; SHA256=$duplicate.SHA256 }
    }
})
$plan | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRoot 'reports/CLEANUP_LOCAL_FACTORS_20261004_PLAN.json') -Encoding utf8
$completed = @()
foreach ($entry in $plan) {
    foreach ($path in @($entry.Path, $entry.Original)) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.SHA256) { throw "Changed input: $path" }
    }
    $temporary = $entry.Path + '.verified-hardlink.tmp'
    if (Test-Path -LiteralPath $temporary) { throw "Temporary path exists: $temporary" }
    New-Item -ItemType HardLink -Path $temporary -Target $entry.Original | Out-Null
    Move-Item -LiteralPath $temporary -Destination $entry.Path -Force
    if ((Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash -ne $entry.SHA256) { throw 'Post-replacement mismatch' }
    $completed += $entry
    $completed | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskRoot 'reports/CLEANUP_LOCAL_FACTORS_20261004_PROGRESS.json') -Encoding utf8
}
foreach ($entry in $entries) {
    if ((Get-FileHash -LiteralPath $entry.Path -Algorithm SHA256).Hash -ne $entry.SHA256) { throw "Final mismatch: $($entry.Path)" }
}
$result = [ordered]@{
    status='verified'; date='2026-10-04'; files_verified=$entries.Count
    hardlinked_files=$completed.Count; logical_reclaimed_bytes=($completed | Measure-Object Bytes -Sum).Sum
    free_before=$beforeFree; free_after=(Get-PSDrive -Name E).Free
    retention='All paths and bytes retained. Immutable historical outputs now share storage; never overwrite these files in place. Compressed backups retained.'
    entries=$completed
}
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath -Encoding utf8
[pscustomobject]$result | Select-Object status,files_verified,hardlinked_files,logical_reclaimed_bytes,free_before,free_after | ConvertTo-Json
