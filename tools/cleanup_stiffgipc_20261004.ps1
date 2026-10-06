param([switch]$Apply)
$ErrorActionPreference='Stop'
$targetRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../stiffGIPC')).Path
if ($targetRoot -ne 'E:\university_class\ComputerGraphics\GIPC\stiffGIPC') { throw 'Unexpected cleanup root' }
$reportRoot=Join-Path $targetRoot 'docs/cleanup_20261004'
$null=New-Item -ItemType Directory -Path $reportRoot -Force
$runs=Join-Path $targetRoot 'benchmarks/stiff4-v1/runs'
$keepPrefixes=@(
 'm23-b-dual-ablation/bunny_cloth_bunny_m/',
 'm23-b-dual-ablation/cloth_hang_l/',
 'm23-b-dual-ablation/paper_fig4_animal_well_o/base/',
 'm23-b-dual-ablation/paper_fig4_animal_well_o/full_b_dual_graph_prototype/',
 'table-conditional-mas-20260927-30f/cloth_table_l/',
 'b-stage23-final-anchor-20260927/cloth_hang_l/'
)
$buildNames=@('build','build-ninja','build-ninja-nofriction','build-local-20260926','build-local-20260926-nofriction','barrier-free/build','barrier-free/build-ninja','barrier-free/build-ninja2')
$remove=[System.Collections.Generic.List[object]]::new()
$retained=[System.Collections.Generic.List[object]]::new()
foreach ($file in Get-ChildItem -LiteralPath $runs -File -Recurse) {
    if ($file.Name -notmatch '\.(f64\w*|i32\w*|u32\w*|bin)$') { continue }
    $relative=[IO.Path]::GetRelativePath($runs,$file.FullName).Replace('\','/')
    $keep=$false
    foreach ($prefix in $keepPrefixes) { if ($relative.StartsWith($prefix)) {$keep=$true;break} }
    $row=[pscustomobject]@{Path=$file.FullName;Bytes=$file.Length;Reason='nonselected experiment raw state'}
    if ($keep) {$retained.Add($row)} else {$remove.Add($row)}
}
foreach ($name in $buildNames) {
    $buildRoot=Join-Path $targetRoot $name
    if (!(Test-Path -LiteralPath $buildRoot)) {continue}
    foreach ($file in Get-ChildItem -LiteralPath $buildRoot -File -Recurse) {
        if ($file.Extension -in @('.obj','.o','.lib','.a','.pdb','.pch','.idb','.ilk','.tlog','.cubin','.fatbin','.ptx','.ii')) {
            $remove.Add([pscustomobject]@{Path=$file.FullName;Bytes=$file.Length;Reason='rebuildable compilation intermediate'})
        }
    }
}
foreach ($row in $remove) {
    $resolved=(Resolve-Path -LiteralPath $row.Path).Path
    if (!$resolved.StartsWith($targetRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) {throw 'Cleanup target escaped root'}
    $item=Get-Item -LiteralPath $resolved
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse point refused'}
    $ancestor=$item.Directory
    while ($ancestor.FullName -ne $targetRoot) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw 'Reparse ancestor refused'}
        $ancestor=$ancestor.Parent
        if ($null -eq $ancestor) {throw 'Missing target ancestor'}
    }
}
$summary=[ordered]@{Root=$targetRoot;DeleteFiles=$remove.Count;DeleteBytes=($remove|Measure-Object Bytes -Sum).Sum;KeepRawFiles=$retained.Count;KeepRawBytes=($retained|Measure-Object Bytes -Sum).Sum;KeepPrefixes=$keepPrefixes;Policy='All non-array experiment files, source, input assets, executables and DLLs preserved; compilation intermediates only in explicit build directories.'}
if (!$Apply) {
    $remove|Export-Csv -LiteralPath (Join-Path $reportRoot 'planned_deletions.csv') -NoTypeInformation -Encoding utf8
    $retained|Export-Csv -LiteralPath (Join-Path $reportRoot 'retained_raw_files.csv') -NoTypeInformation -Encoding utf8
    $summary|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $reportRoot 'plan.json') -Encoding utf8
    $summary|ConvertTo-Json -Depth 4
    exit
}
$resultPath=Join-Path $reportRoot 'result.json'
if (Test-Path -LiteralPath $resultPath) {throw 'Cleanup already recorded'}
$plan=Import-Csv -LiteralPath (Join-Path $reportRoot 'planned_deletions.csv')
if ($plan.Count -ne $remove.Count) {throw 'Cleanup plan changed'}
for ($i=0;$i -lt $plan.Count;$i++) {if ($plan[$i].Path -ne $remove[$i].Path -or [long]$plan[$i].Bytes -ne $remove[$i].Bytes) {throw 'Cleanup plan identity changed'}}
$removeSet=[Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($row in $remove) {$null=$removeSet.Add($row.Path)}
$protected=Get-ChildItem -LiteralPath $runs -File -Recurse|Where-Object {!$removeSet.Contains($_.FullName)}
$before=@{};foreach ($file in $protected) {$before[$file.FullName]=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash}
$before|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $reportRoot 'retained_experiment_sha256.json') -Encoding utf8
$deleted=0L
foreach ($row in $remove) {Remove-Item -LiteralPath $row.Path -Force;$deleted+=$row.Bytes}
foreach ($path in $before.Keys) {if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $before[$path]) {throw 'Retained experiment changed'}}
$summary.DeletedBytes=$deleted;$summary.ProtectedExperimentFilesVerified=$before.Count
$summary.CompletedUtc=[DateTime]::UtcNow.ToString('o')
$summary|ConvertTo-Json -Depth 4|Set-Content -LiteralPath $resultPath -Encoding utf8
$summary|ConvertTo-Json -Depth 4
