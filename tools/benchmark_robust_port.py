"""Serial, resumable local port matrix. Failures remain explicit records."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
ARMS={
    'base':('base',None,'paper'),
    'graph':('base_graph',None,'paper'),
    'paper':('base_toi',None,'paper'),
    'port005_host':('base_toi',.05,'robust'),
    'port005_graph':('base_toi_graph',.05,'robust'),
    'port1_host':('base_toi',1.,'robust'),
    'port1_graph':('base_toi_graph',1.,'robust'),
}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,choices=[2,30,60,100],required=True)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--arms',default='base,graph,port005_host,port005_graph,port1_host,port1_graph')
    parser.add_argument('--substeps',action='store_true')
    parser.add_argument('--audit-friction',action='store_true')
    parser.add_argument('--prefix',default='robust_v32_matched',help='Distinct run identity with the matched Robust movable-DOF stiffness scope')
    args=parser.parse_args()
    arms=args.arms.split(',')
    if any(arm not in ARMS for arm in arms):parser.error('Unknown arm')
    manifest=json.loads((ROOT/'manifests/robust_port_v32.json').read_text())
    for f in manifest['files']:
        assert hashlib.sha256((ROOT/f['path']).read_bytes()).hexdigest()==f['sha256'],f['path']
    for path,identity in manifest['binaries'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==identity['sha256'],path
    records=[]
    for repeat in range(1,args.repeats+1):
        order=arms if repeat%2 else arms[::-1]
        for name in order:
            arm,tol,policy=ARMS[name]
            run_name=f'{args.prefix}_{args.steps}_{name}_r{repeat:02d}'
            if args.substeps:run_name+='_paths'
            run=ROOT/'runs/local'/run_name
            command=[sys.executable,str(ROOT/'tools/run_robust_port.py'),'--arm',arm,
                '--toi-policy',policy,'--scene','cloth_sphere7_l','--steps',str(args.steps),
                '--name',run_name,'--trace','--suite','0','--tol','.01','--pcg-tol','.0001',
                '--dt','.01','--timeout','300']
            if tol is not None:command+=['--robust-velocity-tol',str(tol)]
            if args.substeps:command+=['--substeps']
            if args.audit_friction and policy=='robust':command+=['--audit-friction-snapshot']
            result_file=run/'result.json'
            resumed=result_file.exists()
            if not result_file.exists():
                launcher=ROOT/'runs/local'/f'{run_name}.launcher.log'
                with launcher.open('wb') as stream:
                    status=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
            else:
                requested=json.loads((run/'requested.json').read_text())
                assert requested['steps']==args.steps and requested['arm']==arm and requested['toi_policy']==policy
                assert requested['robust_velocity_tol']==tol and requested['substeps']==args.substeps
                assert requested['audit_friction_snapshot']==(args.audit_friction and policy=='robust')
                expected={'scene':'cloth_sphere7_l','dt':.01,'tol':.01,'pcg_tol':.0001,
                    'suite':'0','trace':True,'platform':'local'}
                for key,value in expected.items():assert requested[key]==value,(run,key)
                assert requested['mu_scope'] is None,run
                binary='local-base' if arm=='base' else 'local-fused-v32'
                assert requested['source_digest']==manifest['source_digest'],run
                assert requested['exe_sha256']==manifest['binaries'][f'builds/{binary}/Release/gipc.exe']['sha256'],run
                assert requested['runner_sha256']==hashlib.sha256((ROOT/'tools/run_robust_port.py').read_bytes()).hexdigest(),run
                status=0 if json.loads(result_file.read_text())['status']=='completed' else 1
            result=json.loads(result_file.read_text()) if result_file.exists() else {'status':'launcher_failed'}
            requested=json.loads((run/'requested.json').read_text()) if (run/'requested.json').exists() else {}
            row={'arm':name,'repeat':repeat,'run':run.relative_to(ROOT).as_posix(),
                'command':command,'resumed':resumed,'source_digest':requested.get('source_digest'),
                'exe_sha256':requested.get('exe_sha256'),'runner_sha256':requested.get('runner_sha256'),
                'launcher_exit_code':status,**result}
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
    report=ROOT/'reports'/f'{args.prefix.upper()}_MATRIX_{args.steps}{"_PATHS" if args.substeps else ""}.json'
    report.write_text(json.dumps({'protocol':vars(args),'runs':records},indent=2)+'\n',encoding='utf-8')
    return int(any(r['launcher_exit_code'] or r['status']!='completed' for r in records))

if __name__=='__main__':raise SystemExit(main())
