"""Freeze a selected, locally tested candidate for a later Linux retest.

No network, remote launch, or solver execution. --inspect only prints the plan.
Packaging is explicit and requires a local report plus the matching active build.
"""
import argparse
import io
import json
from pathlib import Path
import tarfile

from config import ROOT, digest, expand, read, sha

SPEC = ROOT / 'configs/active/autodl_retest.json'
TREES = ('sources/stiff_base', 'sources/stiff_perf_v50',
         'sources/stiff_perf_v54', 'sources/stiff_active',
         'tools/validator', 'references/tight_inclusion')
EXCLUDED = {'.git', '__pycache__', 'build', 'builds', 'Output', 'outputs', '.vs'}
PORTABLE_TOOLS = (
    'tools/active/autodl_prepare.py', 'tools/active/autodl_linux.py',
    'tools/active/autodl_selftest.py', 'tools/active/config.py',
    'tools/active/validate_run.py', 'tools/active/analyze.py',
    'tools/active/cost.py', 'tools/active/compare_costs.py',
    'tools/summarize_cost_trace_active.py',
)


def portable_tools():
    """Explicit portable entry points, including their non-stdlib local imports.

Do not ship Windows build/run wrappers or historical-fixture analyzers without
their data and dependency closure. New tools require deliberate inclusion here.
"""
    return [ROOT / name for name in PORTABLE_TOOLS]


def inventory():
    paths = set()
    for name in TREES:
        base = ROOT / name
        if not base.is_dir():
            raise FileNotFoundError(base)
        paths.update(p for p in base.rglob('*') if p.is_file()
                     and not EXCLUDED.intersection(p.relative_to(base).parts))
    paths.update(portable_tools())
    paths.add(SPEC)
    paths.add(ROOT / 'reports/active/AUTODL_RETEST_PLAN.md')
    if any(p.is_symlink() or not p.resolve().is_relative_to(ROOT) for p in paths):
        raise ValueError('Bundle inventory must contain regular task-local files only')
    return sorted(paths)


def make_plan(candidate, stage, spec=None, triangular_reference=False):
    spec = spec or read(SPEC)
    if stage not in ('smoke', 'window', 'pilot', 'audit', 'paired'):
        raise ValueError('Unknown stage')
    # All four experimental arms inherit exactly the selected execution options.
    selected = expand(candidate)
    selected.update(dt=spec['dt'], trace_velocity=False, diagnostics=[], profile='none',
                    fixed_restrict_study=False, fixed_factor_study=False)
    if selected['mas'] != 'cholesky':
        raise ValueError('Selected candidate must retain stable Cholesky MAS')
    steps = spec['smoke_steps'] if stage == 'smoke' else spec['pilot_steps']
    timeout = spec['smoke_timeout_seconds'] if stage == 'smoke' else spec['pilot_timeout_seconds']
    if stage == 'window':
        timeout = spec['window_timeout_seconds']
    repeats = (range(1, spec['paired_repeats'] + 1) if stage == 'paired' else
               range(1, spec['window_repeats'] + 1) if stage == 'window' else [1])
    runs = []
    for repeat in repeats:
        order = list(spec['arms'])
        if triangular_reference:
            if selected['mas_factor_action'] != 'factor_inverse':
                raise ValueError('Triangular reference is only useful for a factor_inverse candidate')
            order.append('toi_graph_triangular')
        offset = (repeat - 1) % len(order)
        order = order[offset:] + order[:offset]
        if stage == 'audit':
            order = ['stiff', 'toi_graph']
        for key, scene in spec['scenes'].items():
            scene_steps = spec['window_steps'][key] if stage == 'window' else steps
            for arm in order:
                config = (expand({'preset': 'base'}) if arm == 'stiff' else dict(selected))
                if arm != 'stiff':
                    config.update(backend='toi_al' if arm.startswith('toi_') else 'ipc',
                                  execution='conditional_graph' if '_graph' in arm else 'host')
                    if arm == 'toi_graph_triangular':
                        config.update(mas_factor_action='triangular', mas_restrict='warp')
                config.update(scene=scene, steps=scene_steps, dt=spec['dt'], timeout_seconds=timeout,
                              trace_velocity=stage in ('window', 'audit'),
                              diagnostics=['substeps', 'physics'] if stage in ('window', 'audit') else [])
                for tolerance in ('ipc_newton_tol', 'toi_remaining_fraction_tol', 'toi_trial_velocity_tol', 'pcg_rho_tol'):
                    config[tolerance] = selected[tolerance]
                runs.append({'name': f'autodl_{stage}_{key}_{arm}_r{repeat}',
                             'scene_key': key, 'arm': f'{key}:{arm}', 'variant': arm,
                             'repeat': repeat, 'binary': 'base' if arm == 'stiff' else 'active',
                             'config': expand(config)})
    if stage == 'pilot':
        # Freeze three baseline repeats before reviewing candidate quality.
        for repeat in (2, 3):
            for row in list(runs[:]):
                if row['variant'] == 'stiff' and row['repeat'] == 1:
                    runs.append(row | {'name': row['name'].rsplit('_r', 1)[0] + f'_r{repeat}',
                                       'repeat': repeat})
    return {'schema': 1, 'stage': stage, 'candidate_sha256': digest(expand(candidate)),
            'stop_on_failure': False, 'performance_certified': False, 'runs': runs}


def verify_local_build():
    manifest = read(ROOT / 'builds/active/manifest.json')
    for entry in manifest['files']:
        if sha(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('Local source changed since tested build: ' + entry['path'])
    exe = 'builds/active/Release/gipc.exe'
    if sha(ROOT / exe) != manifest['binaries'][exe]['sha256']:
        raise RuntimeError('Active executable identity differs from manifest')
    for entry in read(ROOT / 'manifests/perf_v34.json')['files']:
        if entry['path'].startswith('sources/stiff_base/') and sha(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('Official baseline source differs from frozen reference: ' + entry['path'])
    return manifest


def package(candidate_path, local_report, output, factor_action, triangular_reference=False):
    if output.exists():
        raise FileExistsError(output)
    candidate = read(candidate_path) | {'mas_factor_action': factor_action}
    # Check all stages before writing any artifact.
    plans = {stage: make_plan(candidate, stage, triangular_reference=triangular_reference)
             for stage in ('smoke', 'window', 'pilot', 'audit', 'paired')}
    active = verify_local_build()
    report = local_report.resolve()
    if not report.is_file() or not report.is_relative_to(ROOT):
        raise ValueError('Local report must be an existing task-local file')
    paths = inventory()
    paths.extend(p for p in (report, ROOT / 'builds/active/manifest.json') if p not in paths)
    records = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p),
                'bytes': p.stat().st_size} for p in paths]
    manifest = {'schema': 1, 'files': records, 'source_digest': digest(records),
                'candidate_config': expand(candidate), 'candidate_config_sha256': digest(expand(candidate)),
                'local_report': report.relative_to(ROOT).as_posix(), 'local_report_sha256': sha(report),
                'local_source_digest': active['source_digest'], 'remote_tested': False,
                'triangular_reference': triangular_reference,
                'plan_digests': {stage: digest(plan) for stage, plan in plans.items()}}
    generated = {'autodl_bundle.json': manifest}
    generated.update({f'configs/active/autodl_{s}.json': p for s, p in plans.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'x:gz') as archive:
        for record in records:
            path = ROOT / record['path']
            if sha(path) != record['sha256']:
                raise RuntimeError('Source changed while packaging: ' + str(path))
            archive.add(path, arcname=record['path'], recursive=False)
        for name, value in generated.items():
            data = json.dumps(value, indent=2).encode()
            info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    # A race discovered afterwards invalidates the archive; do not silently replace it.
    if any(sha(ROOT / r['path']) != r['sha256'] for r in records):
        raise RuntimeError('Inputs changed while packaging; archive is invalid and must not be transferred')
    print(json.dumps({'archive': str(output), 'sha256': sha(output), 'bytes': output.stat().st_size,
                      'files': len(records), 'remote_tested': False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-config', required=True, type=Path)
    parser.add_argument('--factor-action', required=True, choices=['triangular', 'factor_inverse'])
    parser.add_argument('--triangular-reference', action='store_true')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--local-report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.inspect:
        candidate = read(args.candidate_config) | {'mas_factor_action': args.factor_action}
        print(json.dumps({s: {'runs': len(make_plan(candidate, s, triangular_reference=args.triangular_reference)['runs']),
                             'max_run_seconds': max(r['config']['timeout_seconds'] for r in make_plan(candidate, s, triangular_reference=args.triangular_reference)['runs'])}
                          for s in ('smoke', 'window', 'pilot', 'audit', 'paired')}, indent=2))
    else:
        if not args.local_report or not args.output:
            parser.error('Packaging requires --local-report and --output; use --inspect for no writes')
        package(args.candidate_config, args.local_report, args.output, args.factor_action, args.triangular_reference)


if __name__ == '__main__':
    main()
