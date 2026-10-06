"""Delete only two locally verified transfer duplicates after all v44 results are downloaded."""
import hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/stiff_toi_cudagraph_20260929').resolve()
target=ROOT/'v44_20261004/reports/CLEANUP_FINAL_V44.json';assert not target.exists()
removed=[];before=shutil.disk_usage(ROOT).free
for relative,expected in [
    ('autodl_perf_v44_20261004.tar.gz','f1350321891aecc044a196d0aff0f0cbf865dc68321c8f09242816e7d365a1ba'),
    ('v44_20261004/results_v44.tar.gz','e1256d31ea966d55ea67e8c0dba6406b3d7aca1cb9d165f3d6cab4c30021678d')]:
    p=ROOT/relative
    assert ROOT in p.resolve(strict=True).parents and not p.is_symlink()
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    removed.append({'path':str(p),'bytes':p.stat().st_size,'sha256':expected})
    p.unlink()
result={'removed_verified_transfer_duplicates':removed,'free_before':before,'free_after':shutil.disk_usage(ROOT).free,
    'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.free','--format=csv,noheader'],text=True).strip(),
    'retained':'Local SHA-verified archives; all remote original sources, binaries, traces, snapshots and reports'}
target.write_text(json.dumps(result,indent=2));print(json.dumps(result))
