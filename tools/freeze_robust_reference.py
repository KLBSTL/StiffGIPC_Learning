"""Read-only snapshot of relevant public Robust and prior reproduction sources."""
import hashlib
import json
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT.parent / 'externals/history-Robust'
DERIVED = ROOT.parent / 'paper_reproduction_20260928/derived/robust_paper_20260929_v3/sources/robust'
DEST = ROOT / 'references/robust_local'
FILES = [
    'LICENSE', 'src/core/core/scene_default_config.cpp',
    'src/backends/cuda/engine/advance_al.cu',
    'src/backends/cuda/active_set_system/global_active_set_manager.cu',
    'src/backends/cuda/affine_body/abd_al_stiffness_estimator.cu',
    'src/backends/cuda/finite_element/fem_al_stiffness_estimator.cu',
    'src/backends/cuda/affine_body/abd_linear_subsystem.cu',
    'src/backends/cuda/finite_element/fem_linear_subsystem.cu',
    'src/backends/cuda/linear_system/global_linear_system.cu',
    'src/backends/cuda/linear_system/linear_fused_pcg.cu',
    'src/backends/cuda/affine_body/abd_active_set_reporter.cu',
    'src/backends/cuda/finite_element/fem_active_set_reporter.cu',
    'src/backends/cuda/contact_system/al_contact_function.h',
    'src/backends/cuda/collision_detection/filters/al_vertex_half_plane_trajectory_filter.cu',
    'src/backends/cuda/collision_detection/filters/stackless_bvh_simplex_trajectory_filter.cu',
]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--extend',action='store_true')
    args=parser.parse_args()
    if DEST.exists() and not args.extend:
        raise RuntimeError('Reference snapshot already exists; preserve it.')
    records = []
    for source_root, group, names in [(PUBLIC, 'public', FILES),
            (DERIVED, 'prior_derived', ['advance_al_public.cu', 'advance_al_paper.cu'])]:
        for name in names:
            source = source_root / name
            data = source.read_bytes()
            target = DEST / group / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target.read_bytes())!=digest(data):
                raise RuntimeError(f'Existing snapshot differs; create a new version: {target}')
            if not target.exists():target.write_bytes(data)
            if digest(source.read_bytes()) != digest(data):
                raise RuntimeError(f'Source changed while reading: {source}')
            records.append({'path':target.relative_to(ROOT).as_posix(),
                            'source':str(source), 'sha256':digest(data), 'bytes':len(data)})
    commit = subprocess.check_output(['git', '-C', str(PUBLIC), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(PUBLIC), 'status', '--short'], text=True).strip()
    manifest = {'public_commit':commit, 'public_worktree_status':dirty,
                'scope':'Selected public files plus separately labeled prior derived variants; donors were only read.',
                'files':records}
    (DEST/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'files':len(records), 'public_commit':commit, 'destination':str(DEST)}))

if __name__ == '__main__':
    main()
