"""Bind the completed native-history representative gate without rewriting stage 1."""
import ast
import json
from pathlib import Path
import re
import shutil
import sys

from m3_numeric import ROOT, need
import numpy as np
import scipy
from local_identity import record, verify, tree
from linux_runner import write_new
# Import the actual entry modules to collect their local Python dependencies.
# Their workloads are guarded by __main__.
import run_native_history
import analyze_native_history
import audit_native_history_representative

RAW = ROOT / 'runs/innovation_p5_native_history_20261016'
REPORT = ROOT / 'reports/innovation_p5_native_history_20261016'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    destination = REPORT / 'REPRESENTATIVE_SEAL.json'
    need(not destination.exists(), 'Representative stage already sealed')
    parent_path = REPORT / 'BUILD_AND_SMOKE_SEAL.json'
    parent = read(parent_path)
    verify(parent['files'])
    # Also verify the older QR-free parent, using stage 1's explicit old-byte mapping.
    older = read(parent['parent']['path'])
    mapping = {r['original']['path']: r for r in parent['parent_mutable_mapping']}
    for row in older['files']:
        if row['path'] in mapping:
            replacement = mapping[row['path']]
            need(replacement['original'] == row, 'Parent mapping mismatch')
            verify([replacement['archive']])
        else:
            verify([row])
    packet = read(RAW / 'seal.json')
    identity = packet['identity']
    verify(identity['sources'] + [identity['exe']] + identity['dlls']
           + [packet['runner'], packet['protocol']])
    ledger = read(RAW / 'representative_ledger.json')
    analysis = read(RAW / 'analysis/summary.json')
    audit = read(REPORT / 'REPRESENTATIVE_AUDIT.json')
    need(ledger['complete'] and len(ledger['runs']) == 2, 'Incomplete native ledger')
    need(analysis['implementation_checks_passed'] and audit['passed'], 'Numerical audit incomplete')
    verify(analysis['inputs'] + [analysis['analyzer']] + audit['bindings'] + [audit['source']])
    for task in ledger['runs']:
        need(task['passed'] and task['result']['cleanup_owned_job_empty'], 'Native Job/validation failed')
        verify(task['studies'])
    need(len(audit['native_solutions']) == 24 and len(audit['banks']) == 12
         and len(audit['target_solutions']) == 14, 'Audit counts changed')
    for row in analysis['methods']:
        verify([row['solution']])
    for gate in analysis['gates']:
        rows = [r for r in analysis['methods'] if r['scene'] == gate['scene'] and r['pair'] == gate['pair']]
        direct = next(r for r in rows if r['method'] == 'direct_native_history')
        stats = next(r for r in rows if r['method'] == 'statistics_native_history')
        passed = bool(direct['converged'] and stats['converged']
                      and stats['iterations'] <= 1.1 * direct['iterations'])
        need(gate['direct_iterations'] == direct['iterations']
             and gate['statistics_iterations'] == stats['iterations'] and gate['passed'] == passed,
             'Iteration gate disagrees with actual methods')
    primary_passed = all(g['passed'] for g in analysis['gates'] if g['primary'])
    need(primary_passed == analysis['primary_iteration_gate_passed'], 'Primary gate mismatch')
    for name in ('CPU_methods', 'CPU_curves', 'source_quality', 'native_solutions', 'iteration_gate'):
        source = RAW / 'analysis' / (name + '.csv')
        target = REPORT / 'representative_tables' / source.name
        with source.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
        need(record(source)['sha256'] == record(target)['sha256'], 'Table copy mismatch')
    sources = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'tools'):
                sources.add(path)
    bindings = []
    for path in sorted(sources):
        ast.parse(path.read_text(encoding='utf-8-sig'))
        target = REPORT / 'sources_representative' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as inp, target.open('xb') as out:
            shutil.copyfileobj(inp, out)
        current, copy = record(path), record(target)
        need((current['bytes'], current['sha256']) == (copy['bytes'], copy['sha256']), 'Source copy mismatch')
        bindings.append(dict(source=current, snapshot=copy))
    report = REPORT / 'REPRESENTATIVE_REPORT.md'
    for link in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        need(link.startswith('https://') or (report.parent / link).exists()
             or report.parent / link == destination, 'Missing representative report link')
    records = parent['files'] + tree(RAW) + tree(REPORT)
    records += [b['source'] for b in bindings]
    records += [record(p) for p in (ROOT / 'build').glob('native_history*.log')
                if 'seal' not in p.name]
    unique = {r['path']: r for r in records}
    files = [unique[k] for k in sorted(unique)]
    verify(files)
    write_new(destination, dict(schema='native_history_representative.v1', files=files,
        parent=record(parent_path), parent_files_verified=len(parent['files']),
        older_parent_files_verified=len(older['files']), older_mutable_mappings=len(mapping),
        source_bindings=bindings, binary=identity['exe'], Python=sys.version,
        NumPy=np.__version__, SciPy=scipy.__version__, representative_validation_pending=False,
        native_systems=4, native_solves=24, native_banks=12, target_solutions=14,
        primary_iteration_gate_passed=primary_passed, all_repetition_gates_passed=all(g['passed'] for g in analysis['gates']),
        native_rank8_source_CPU_reference_transfer_validated=primary_passed,
        native_Graph_history_captured=False, native_balanced_PCG_implemented=False,
        production_threshold_changed=False, performance_certified=False,
        stage1_pending_is_historical_snapshot=True,
        next_finite_gate='protected native fixed-A/b balanced PCG correctness and complete cost accounting'))
    verify(read(destination)['files'])
    print(json.dumps(dict(sealed=True, files=len(files), source_modules=len(bindings),
        native_solves=24, native_banks=12, target_solutions=14,
        primary_iteration_gate_passed=primary_passed, performance_certified=False)), flush=True)


if __name__ == '__main__':
    main()
