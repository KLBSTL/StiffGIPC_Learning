"""Bounded measurement of restart equivalence before exit-sufficiency ablation."""
import json,subprocess,sys,hashlib,itertools
from pathlib import Path
import numpy as np
from analyze_perf_v36 import matrix_from_snapshot
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def difference(x,y):
    delta=x-y;maximum=float(np.max(np.abs(delta))) if x.size else 0.
    return {'max_abs':maximum,'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(x),1e-30)),
            'scale':float(np.max(np.abs(x))) if x.size else 0.}
def gate(names,frame):
    checks=[]
    for an,bn in itertools.combinations(names,2):
        a=ROOT/'runs/local'/an;b=ROOT/'runs/local'/bn;buffers=[]
        for field in ['vertices','old_vertices','velocities','x_tilde','abd_q','abd_q_prev','abd_q_tilde','abd_q_v']:
            x=np.fromfile(a/f'final_checkpoint/{field}.bin',dtype='<f8');y=np.fromfile(b/f'final_checkpoint/{field}.bin',dtype='<f8')
            assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            d=difference(x,y);buffers.append({'buffer':field,**d,'passed':d['max_abs']<=1e-8 and d['relative_l2']<=1e-8})
        _,A,rhs=matrix_from_snapshot(a/f'fixed/f{frame}_n1');_,B,other=matrix_from_snapshot(b/f'fixed/f{frame}_n1')
        assert A.shape==B.shape
        A=A.tocsr();B=B.tocsr();A.sum_duplicates();B.sum_duplicates();delta=A-B
        da={'max_abs':float(np.max(np.abs(delta.data))) if delta.nnz else 0.,'relative_l2':float(np.linalg.norm(delta.data)/max(np.linalg.norm(A.data),1e-30)), 'scale':float(np.max(np.abs(A.data)))}
        db=difference(rhs,other)
        for d in [da,db]:d['passed']=d['relative_l2']<=1e-10 and d['max_abs']<=1e-10*(1+d['scale'])
        checks.append({'a':an,'b':bn,'buffers':buffers,'A':da,'b_rhs':db,'passed':all(d['passed'] for d in buffers+[da,db])})
    restore=[]
    for name in names[1:]:
        run=ROOT/'runs/local'/name;rr=read(run/'output/checkpoint_restore.json');assert rr['device_bytes_verified'];restore.append(rr)
        assert (run/'trace/state_0000.bin').read_bytes()==(ROOT/'runs/local'/names[0]/f'trace/state_{frame-1:04d}.bin').read_bytes()
    for name in names:
        study=read(ROOT/'runs/local'/name/f'fixed/f{frame}_n1_study.json')
        assert study['system_unchanged'] and study['primary_restored_bitwise']
    return {'passed':all(c['passed'] for c in checks),'comparisons':checks,'restore':restore}
def main():
    target=ROOT/'reports/V54_EXIT_MATRIX.json';assert not target.exists();rows=[]
    manifest=read(ROOT/'manifests/perf_v54_local.json')
    for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
    for path,e in manifest['binaries'].items():assert sha(ROOT/path)==e['sha256']
    old=read(ROOT/'reports/V54_CLOTH_MATRIX.json')
    def run(scene,label,frame,kind,repeat=1,checkpoint=None):
        template=next(r for r in old if r['scene']==scene and r['variant']=='guard' and r['repeat']==1)
        name=f'v54_exit_{label}_{kind}_r{repeat}';cmd=template['command'].copy();steps=1 if checkpoint else frame
        mode='velocity_only' if kind in ['velocity','fromzero'] else 'native'
        for key,val in [('--name',name),('--steps',str(steps)),('--timeout','90'),('--inner-exit',mode)]:cmd[cmd.index(key)+1]=val
        cmd+=['--checkpoint-final','--fixed-study','--compact-study','--fixed-study-frames',str(frame),'--fixed-study-directions','1',
              '--stage-from-frame',str(frame),'--linear-stages']
        if checkpoint:cmd+=['--checkpoint-load',str(checkpoint)]
        elif kind=='seed':cmd+=['--checkpoint-frame',str(frame)]
        with (ROOT/'reports'/f'{name}_runner.log').open('x') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        result=read(ROOT/'runs/local'/name/'result.json');row={**template,'name':name,'kind':kind,'repeat':repeat,'target_frame':frame,
             'inner_exit':mode,'result':result,'command':cmd,'returncode':p.returncode}
        rows.append(row);target.write_text(json.dumps(rows,indent=2));print(json.dumps({'name':name,'result':result}),flush=True);return row
    for label,scene,frame in [('fixed','cloth_fixed_bunny_l',24),('sphere','cloth_sphere7_l',25)]:
        seed=run(scene,label,frame,'seed')
        if seed['result']['status']!='completed':continue
        cp=ROOT/'runs/local'/seed['name']/'checkpoint';names=[seed['name']]
        for rep in [1,2]:
            row=run(scene,label,frame,'resume',rep,cp)
            if row['result']['status']!='completed':break
            names.append(row['name'])
        g=gate(names,frame) if len(names)==3 else {'passed':False,'reason':'required native resume failed'}
        (ROOT/'reports'/f'V54_EXIT_GATE_{label}.json').write_text(json.dumps(g,indent=2));print(json.dumps({'gate':label,'passed':g['passed']}),flush=True)
        for rep in [1,2]:
            row=run(scene,label,frame,'velocity' if g['passed'] else 'fromzero',rep,cp if g['passed'] else None)
            if row['result']['status']!='completed':break
if __name__=='__main__':main()
