"""Summarize isolated AutoDL diagnostics without copying large traces or logs."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NAMES = [
    'autodl_v18_mass_bunny_host100',
    'autodl_v18_mass10_bunny_host100',
    'autodl_v18_mass2_bunny_host100',
    'autodl_v18_mass1_strict_volume_bunny100',
    'autodl_v18_mass1_strict_volume_limit24m_bunny100',
]


def main():
    rows = []
    for name in NAMES:
        run = ROOT / 'runs/autodl' / name
        requested = json.loads((run / 'requested.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        messages = [line.strip() for line in (run / 'run.log').read_text(errors='replace').splitlines()
                    if 'GIPC controlled failure:' in line or 'Growing CCD pair storage' in line]
        rows.append({
            'run': name, 'mu_mode': requested['mu_mode'], 'mu_scale': requested['mu_scale'],
            'safe_injectivity': requested['safe_injectivity'],
            'ccd_pair_limit': requested['ccd_pair_limit'],
            'status': result['status'], 'recorded_frames': result.get('recorded_frames'),
            'solver_seconds': result.get('solver_seconds'),
            'last_controlled_failure': next((line for line in reversed(messages)
                                             if 'GIPC controlled failure:' in line), None),
            'last_pair_growth': next((line for line in reversed(messages)
                                      if 'Growing CCD pair storage' in line), None),
        })
    output = ROOT / 'reports/autodl_v18_mass_ablation_summary.json'
    output.write_text(json.dumps({'scope': 'AutoDL CUDA 12.8 single-run diagnostics',
                                  'runs': rows}, indent=2), encoding='utf-8')
    print(json.dumps(rows))


if __name__ == '__main__':
    main()
