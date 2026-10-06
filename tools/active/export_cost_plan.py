"""Export a frozen finite plan's completed Nsight captures; CPU only.

Outputs are exclusive: rerunning cannot silently replace a capture or analysis.
The analyzer keeps inclusive waits and unattributed graph nodes separate.
"""
import argparse
import json
import subprocess
import sys
from config import ROOT, read, sha

NSYS = 'C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe'


def command(argv, log):
    with log.open('xb') as output:
        output.write((json.dumps(argv) + '\n').encode())
        result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=120)
    if result.returncode:
        raise RuntimeError(f'Export failed ({result.returncode}): {log}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    path = ROOT / args.plan
    plan = read(path)
    for task in plan['runs']:
        cfg = read(ROOT / task['config']) if isinstance(task['config'], str) else task['config']
        if cfg.get('profile', 'none') == 'none':
            continue
        run = ROOT / 'runs/active' / task['name']
        if read(run / 'result.json')['status'] != 'completed':
            raise RuntimeError(f'Incomplete capture: {run}')
        database = run / 'nsight.sqlite'
        analysis = run / 'activity_analysis.json'
        if database.exists() or analysis.exists():
            raise FileExistsError(f'Export already exists: {run}')
        command([NSYS, 'export', '--type=sqlite', '--output=' + str(database), str(run / 'nsight.nsys-rep')], run / 'export.log')
        command([sys.executable, str(ROOT / 'tools/active/analyze_ipc_light_cost.py'), '--sqlite', str(database), '--output', str(analysis), '--top', '40'], run / 'analysis.log')
        with (run / 'cost_export_identity.json').open('x', encoding='utf-8') as stream:
            json.dump({'plan_sha256': sha(path), 'capture_sha256': sha(run / 'nsight.nsys-rep'),
                       'exporter_sha256': sha(__file__), 'analyzer_sha256': sha(ROOT / 'tools/active/analyze_ipc_light_cost.py')}, stream)
        print(json.dumps({'run': run.name, 'exported': True}), flush=True)


if __name__ == '__main__':
    main()
