"""Byte-verified hard links preserve every result path and content."""
import hashlib,json,os,shutil
from pathlib import Path
r=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929/v42a_20261003').resolve()
report=r/'reports/factor_dedup.json';assert not report.exists();rows=[];before=shutil.disk_usage(r).free
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for case in ['v39_default','v39_strict','v41_graph','v41_host','v41_wide']:
    original=r/'runs'/f'{case}_m6/factors.bin';digest=sha(original)
    for path in [r/'runs'/f'{case}_m7/factors.bin',r/'strict16'/case/'factors.bin']:
        assert r in original.resolve(strict=True).parents and r in path.resolve(strict=True).parents
        assert not path.is_symlink() and not original.is_symlink() and sha(path)==digest
        temp=path.with_name('factors.verified-hardlink.tmp');assert not temp.exists()
        os.link(original,temp);os.replace(temp,path);assert sha(path)==digest
        rows.append({'path':str(path),'shared_with':str(original),'sha256':digest,'bytes':path.stat().st_size})
report.write_text(json.dumps({'files':rows,'free_before':before,'free_after':shutil.disk_usage(r).free},indent=2));print(report.read_text())
