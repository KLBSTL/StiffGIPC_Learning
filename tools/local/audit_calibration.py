"""Freeze observation bounds from complete original-Stiff repetitions only."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
sys.path.insert(0, str(ROOT/'tools/local'))
from config import read
from linux_runner import require, write_new
from local_identity import record
from pool_metrics import state_comparison
from quality_analysis import input_comparison


def main():
    session = ROOT/'runs/audit_quality_reference_120'
    ledger = read(session/'ledger.json')
    require(not ledger['skipped'] and len(ledger['runs']) == 9, 'Incomplete three-scene calibration')
    result = dict(schema='audit_reference_observation.v1', ledger=record(session/'ledger.json'),
        identity=record(session/'identity.json'), scenes={},
        numerical_stretch_floor=1e-6, numerical_J_floor=1e-6,
        fixed_drift_floor_m=1e-10, performance_certified=False, quality_certified=False,
        physical_budget='No extra material deformation budget. Floors only for numerical comparison.',
        limitations=['Original Stiff has no actual velocity export.',
            'Observed baseline inversions are recorded; passing these bounds does not establish physical safety.',
            'No independent accepted-path CCD certificate. Shared WDDM timing is diagnostic.'])
    for scene in sorted({r['scene'] for r in ledger['runs']}):
        rows = [r for r in ledger['runs'] if r['scene'] == scene]
        require(len(rows) == 3 and all(r['arm'] == 'base' and r['summary']['hard_checks_passed']
            and r['summary']['frames'] == 120 for r in rows), 'Not complete original-Stiff calibration')
        summaries = [r['summary'] for r in rows]
        bounds = {key:max(s[key] for s in summaries)+1e-6 for key in ('max_stretch','p99_stretch')}
        bounds['fixed_drift_m'] = max(s['fixed_drift_m'] for s in summaries)+1e-10
        minimum = [s['fem_min_J'] for s in summaries if s['fem_min_J'] is not None]
        bounds['fem_min_J'] = min(minimum)-1e-6 if minimum else None
        bounds['fem_nonpositive'] = max(s['fem_nonpositive'] for s in summaries)
        pairs = []
        for a in range(len(rows)):
            for b in range(a+1, len(rows)):
                left, right = session/rows[a]['name'], session/rows[b]['name']
                matching = input_comparison(left, right)
                require(matching['passed'], 'Baseline initial inputs differ')
                pairs.append(dict(left=rows[a]['name'], right=rows[b]['name'],
                    initial_input=matching, trajectory=state_comparison(left, right)))
        result['scenes'][scene] = dict(bounds=bounds,
            runs=[dict(name=r['name'], **r['summary']) for r in rows],
            baseline_pair_comparisons=pairs)
    destination = ROOT/'reports/AUDIT_REFERENCE_OBSERVATION.json'
    write_new(destination, result)
    print(json.dumps(dict(path=str(destination), scenes={k:v['bounds'] for k,v in result['scenes'].items()})))


if __name__ == '__main__':
    main()
