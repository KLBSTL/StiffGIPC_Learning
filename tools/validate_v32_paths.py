"""Independent CPU audit of every completed v32 accepted-path pilot."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def validate(entry):
    run=ROOT/entry['run']
    report=ROOT/'reports'/f'v32_matched_{entry["arm"]}_accepted_ccd.json'
    executable=ROOT/'builds/validator/Release/validate_path.exe'
    command=[str(executable),str(run/'trace'),str(report),'substeps','--stable-nh1']
    if report.exists():raise RuntimeError(f'Preserve existing validation: {report}')
    with (ROOT/'builds'/f'v32_matched_{entry["arm"]}_ccd.log').open('wb') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    return {'arm':entry['arm'],'run':entry['run'],'command':command,'exit_code':result.returncode,
        'validator_sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),
        'validation':read(report) if report.exists() else None}

def main():
    matrix=read(ROOT/'reports/ROBUST_V32_MATCHED_MATRIX_60_PATHS.json')
    tasks=[e for e in matrix['runs'] if e['status']=='completed']
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(validate,e) for e in tasks]):
            row=future.result();rows.append(row);audit=row['validation'] or {}
            print(json.dumps({'arm':row['arm'],'passed':audit.get('passed'),
                'paths':audit.get('paths_checked'),'flags':audit.get('conservative_collision_flags')}),flush=True)
    output={'protocol':'60 physical frames, all accepted substeps, Stable NH1, at most two CPU jobs',
        'runs':rows,'all_passed':all(r['validation'] and r['validation'].get('passed') for r in rows)}
    (ROOT/'reports/ROBUST_V32_MATCHED_PATH_VALIDATION.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    return 0 if output['all_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
