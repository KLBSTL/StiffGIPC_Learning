"""Bounded integrated component, guard and historical MAS correctness gates."""
import argparse
import json
import os
import subprocess
from config import ROOT, sha, read

CASES = [('smoke',37,'mas_smoke','initial'),('default',39,'wide_bunny','failure'),
         ('strict',39,'wide_bunny_strict','failure'),('v41_graph',41,'double_bunny','failure'),
         ('v41_host',41,'double_bunny_host','failure'),('v41_wide',41,'wide_guard','failure')]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--name', required=True)
    p.add_argument('--mas-restrict',choices=['serial','warp'],default='serial')
    p.add_argument('--mas-factor-action',choices=['triangular','factor_inverse'],default='factor_inverse')
    p.add_argument('--eligibility',action='store_true',help='Run independent BVH summary GPU fixture')
    p.add_argument('--bounded-ccd',action='store_true',help='Run independent request-bounded CCD GPU fixture')
    p.add_argument('--contact-pool',action='store_true',help='Run swept candidate pool GPU fixture')
    a=p.parse_args()
    if any(x in a.name for x in ('/', '\\', ':')) or a.name in ('.','..'):
        raise ValueError('Single directory name required')
    out=ROOT/'runs/active'/a.name; out.mkdir()
    exe=ROOT/'builds/active/Release/gipc.exe'
    manifest=read(ROOT/'builds/active/manifest.json')
    assert sha(exe)==manifest['binaries'][exe.relative_to(ROOT).as_posix()]['sha256']
    tasks=[('components',{'GIPC_VALIDATE_COMPONENTS':str(out/'components.json')},out/'components.json'),
           ('guards',{'GIPC_PCG_GUARD_FIXTURE':str(out/'guards.json')},out/'guards.json')]
    if a.eligibility:
        tasks.insert(0,('bvh_eligibility',{'GIPC_BVH_ELIGIBILITY_FIXTURE':str(out/'bvh_eligibility.json')},out/'bvh_eligibility.json'))
    if a.bounded_ccd:
        tasks.insert(0,('bounded_ccd',{'GIPC_BOUNDED_CCD_FIXTURE':str(out/'bounded_ccd.json')},out/'bounded_ccd.json'))
    if a.contact_pool:
        tasks.insert(0,('contact_pool',{'GIPC_CONTACT_POOL_FIXTURE':str(out/'contact_pool.json'),
            'GIPC_CONTACT_POOL':'1','GIPC_CONTACT_POOL_VALIDATE':'1'},out/'contact_pool.json'))
    for name,v,folder,kind in CASES:
        prefix=ROOT/f'downloads/autodl_perf_v{v}_20261003/runs/autodl/autodl_perf_v{v}_{folder}/mas_audit_{kind}'
        tasks.append((name,{'GIPC_MAS_CHOLESKY_FIXTURE':str(prefix),
                           'GIPC_MAS_FIXTURE_OUTPUT':str(out/name)},out/name/'fixture.json'))
    result={'exe_sha256':sha(exe),'mas_restrict':a.mas_restrict,'mas_factor_action':a.mas_factor_action,'checks':[],'passed':True}
    lock=ROOT/'runs/active/.gpu.lock'
    with lock.open('x') as f: f.write('fixtures:'+a.name)
    try:
        for name,settings,path in tasks:
            env={k:v for k,v in os.environ.items() if not k.startswith('GIPC_')};env.update(settings)
            env['GIPC_MAS_RESTRICT_MODE']=a.mas_restrict
            env['GIPC_MAS_FACTOR_ACTION']=a.mas_factor_action
            with (out/(name+'.log')).open('wb') as log:
                try:
                    r=subprocess.run([str(exe)],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                    code=r.returncode
                except subprocess.TimeoutExpired:
                    code='timeout'
            data=read(path) if path.exists() else {}
            passed=code==0 and bool(data) and data.get('passed',data.get('pass',True))
            result['checks'].append({'case':name,'exit_code':code,'passed':passed,'result':data})
            result['passed'] &= passed
            (out/'result.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({'case':name,'passed':passed,'exit_code':code}),flush=True)
    finally:
        lock.unlink()
    return int(not result['passed'])

if __name__=='__main__': raise SystemExit(main())
