"""Offline raw-data aggregation. Incomplete/failed rows never become timing data."""
import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path
import sys
import numpy as np
from protocol import ROOT,SCENES,ARMS,CHECKPOINTS,read,write,sha,require,plan
sys.path.insert(0,str(ROOT/'tools/bench'))
from pool_metrics import export_evidence,metrics


def observation(folder,task,result):
    require(result['status']=='completed','Non-complete observation')
    frames=read(folder/'output/stats.json')['frames']
    with (folder/'trace/frames.csv').open() as f:times=list(csv.DictReader(f))
    require(len(frames)==len(times)==task['config']['steps'],'Missing physical frames')
    require(export_evidence(folder,len(frames),'state')['passed'],'Invalid position export')
    require(not any(f.get('newton_exit')=='iteration_limit' for f in frames),'Nonlinear limit')
    pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    require(pcg and not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg),'PCG failed')
    require(all(all(k in f.get('phase_ms',{}) for k in ('assembly','pcg','ccd','line_search')) for f in frames),'Missing phase timings')
    totals={k:sum(f['phase_ms'][k] for f in frames)/1000 for k in ('assembly','pcg','ccd','line_search')}
    elapsed=sum(float(t['solver_ms']) for t in times)/1000
    require(elapsed>0 and all(np.isfinite(v) and v>=0 for v in totals.values()),'Invalid timing')
    remainder=elapsed-sum(totals.values())
    require(remainder>=-1e-3*max(1,elapsed),'Phase times exceed synchronized host total')
    count=sum(p['iterations'] for p in pcg);n=len(pcg)
    q=metrics(folder)
    return dict(Hess=totals['assembly'],LSolver=totals['pcg'],CCD=totals['ccd'],LineS=totals['line_search'],
        Misc=remainder,TimeTot=elapsed,P_bar=count/n,N_bar=n/len(frames),P_tot=count,N_tot=n,
        ms_per_step=1000*elapsed/len(frames),quality={k:q[k] for k in ('max_stretch','p99_stretch','fixed_drift_m','finite')},
        raw_files={name:sha(folder/name) for name in ('requested.json','result.json','output/stats.json','trace/frames.csv')})


def physical_inputs_equal(a,b):
    names=['topology.bin','masses.bin','boundary_types.bin','body_ids.bin','state_0000.bin','metadata.json']
    evidence={n:sha(a/'trace'/n)==sha(b/'trace'/n) for n in names}
    # Runtime solver tolerances deliberately differ; compare the frozen scene minus run overrides.
    x=read(a/'output/scene.json');y=read(b/'output/scene.json')
    x.pop('effective_run',None);y.pop('effective_run',None);evidence['frozen_scene']=x==y
    if all((p/'trace/velocity_0000.bin').exists() for p in (a,b)):
        evidence['initial_actual_velocity']=sha(a/'trace/velocity_0000.bin')==sha(b/'trace/velocity_0000.bin')
    return dict(passed=all(evidence.values()),checks=evidence)


def position_error(candidate,reference):
    a=np.fromfile(candidate,dtype='<f8');b=np.fromfile(reference,dtype='<f8')
    require(a.shape==b.shape and a.size and a.size%3==0 and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid paired state')
    delta=(a-b).reshape(-1,3);norm=np.linalg.norm(delta,axis=1)
    return dict(rms_mm=1000*float(np.sqrt(np.mean(norm**2))),max_mm=1000*float(norm.max()))


def checkpoint_gate(session,ledger):
    if not ledger['complete']:
        return dict(passed=False,checks=[],scope='Incomplete matched stage; no equivalence or ablation claim')
    checks=[];inputs=[]
    for scene,frame in CHECKPOINTS.items():
        seed=session/f'seed_{scene.lower()}_original_r1'
        cp=seed/'checkpoint';meta=read(cp/'checkpoint.json')
        for r in ledger['runs']:
            t=r['task']
            if t['scene_key']!=scene:continue
            out=session/t['name'];restore=read(out/'output/checkpoint_restore.json')
            request=read(out/'requested.json')
            exact={name:(cp/(name+'.bin')).read_bytes()==(out/'trace'/trace).read_bytes()
                for name,trace in (('vertices','state_0000.bin'),('velocities','velocity_0000.bin'))}
            exact.update(binary=restore['binary_id']==meta['binary_id']==request['exe_sha256'],
                frame=restore['completed_physical_frames']==meta['completed_physical_frames']==frame-1,
                all_device_buffers=restore['device_bytes_verified'] and restore['restored_buffers']==len(meta['buffers']),
                physical_context=restore['context_verified'])
            inputs.append(dict(scene=scene,run=t['name'],checks=exact,passed=all(exact.values())))
            if t['arm']!='original':continue
            comparisons={}
            for kind in ('state','velocity'):
                a=np.fromfile(seed/f'trace/{kind}_{frame:04d}.bin',dtype='<f8')
                b=np.fromfile(out/f'trace/{kind}_0001.bin',dtype='<f8')
                require(a.shape==b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid continuation')
                maximum=float(np.abs(a-b).max());relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-30))
                comparisons[kind]=dict(max_abs=maximum,relative_l2=relative,passed=maximum<=1e-8 and relative<=1e-8)
            checks.append(dict(scene=scene,run=t['name'],restore=restore,comparisons=comparisons,
                passed=restore['device_bytes_verified'] and restore['context_verified'] and all(x['passed'] for x in comparisons.values())))
    return dict(passed=ledger['complete'] and len(inputs)==24 and all(r['passed'] for r in inputs),
        input_checks=inputs,checks=checks,hot_cold_1e8_passed=len(checks)==6 and all(r['passed'] for r in checks),
        scope='Ablation input gate: same binary, frame, exact initial position/velocity, every device buffer and physical context restored. '
              'Hot/cold endpoint differences and the failed 1e-8 continuity test are separately retained; '
              'cold Graph/MAS/BVH reconstruction is not certified equivalent to hot caches. '
              'All ablation rows use cold reconstruction; no physical-quality certification.')


def tables(session,output):
    output.mkdir(parents=True,exist_ok=False);summary=dict(plan=plan(),performance=[],graph=[],matched=[],missing=[],
        physical_quality_certified=False,performance_certified=False)
    perf=session/'performance_ledger.json'
    if perf.exists():
        ledger=read(perf);lookup={}
        for r in ledger['runs']:
            if r['result']['status']!='completed':summary['missing'].append(dict(name=r['task']['name'],status=r['result']['status']));continue
            t=r['task'];lookup[t['scene_key'],t['arm'],t['repeat']]=observation(session/t['name'],t,r['result'])
        for scene in SCENES:
            for arm in ARMS:
                rows=[lookup.get((scene,arm,r)) for r in (1,2,3)]
                if any(r is None for r in rows):summary['missing'].append(dict(scene=scene,arm=arm,reason='three complete repeats required'));continue
                group=dict(scene=scene,method=arm,**{k:statistics.median(r[k] for r in rows) for k in
                    ('Hess','LSolver','CCD','LineS','Misc','TimeTot','P_bar','N_bar','P_tot','ms_per_step')},raw=rows)
                if arm in ('stiff','ours'):
                    group['S']=1. if arm=='stiff' else None
                    if arm=='ours' and all((scene,'stiff',r) in lookup for r in (1,2,3)):
                        comparisons=[physical_inputs_equal(session/f'performance_{scene.lower()}_stiff_r{r}',
                            session/f'performance_{scene.lower()}_ours_r{r}') for r in (1,2,3)]
                        require(all(c['passed'] for c in comparisons),'Physical input differs in main pair')
                        group['S']=statistics.median(lookup[scene,'stiff',r]['TimeTot']/lookup[scene,'ours',r]['TimeTot'] for r in (1,2,3))
                        group['input_checks']=comparisons
                    summary['performance'].append(group)
            if all((scene,a,r) in lookup for a in ('host','graph') for r in (1,2,3)):
                comparisons=[physical_inputs_equal(session/f'performance_{scene.lower()}_host_r{r}',
                    session/f'performance_{scene.lower()}_graph_r{r}') for r in (1,2,3)]
                require(all(c['passed'] for c in comparisons),'Graph pair physical input differs')
                summary['graph'].append(dict(scene=scene,
                    **{a:{k:statistics.median(lookup[scene,a,r][k] for r in (1,2,3)) for k in ('ms_per_step','N_bar','P_bar')} for a in ('host','graph')},
                    T_B_over_T_G=statistics.median(lookup[scene,'host',r]['TimeTot']/lookup[scene,'graph',r]['TimeTot'] for r in (1,2,3)),input_checks=comparisons))
    matched=session/'matched_ledger.json'
    if matched.exists():
        ledger=read(matched);gate=checkpoint_gate(session,ledger)
        if not (session/'checkpoint_input_gate.json').exists():write(session/'checkpoint_input_gate.json',gate)
        else:require(read(session/'checkpoint_input_gate.json')==gate,'Prior checkpoint input gate differs')
        summary['checkpoint_gate']=gate
        if gate['passed']:
            for scene,frame in CHECKPOINTS.items():
                for arm in ('original','pure','ours'):
                    selected=[r for r in ledger['runs'] if r['task']['scene_key']==scene and r['task']['arm']==arm]
                    require(len(selected)==3 and all(r['result']['status']=='completed' for r in selected),'Incomplete checkpoint candidates')
                    obs=[observation(session/r['task']['name'],r['task'],r['result']) for r in selected]
                    comparisons=[]
                    for c,ref in itertools.product((1,2,3),(1,2,3)):
                        a=session/f'matched_{scene.lower()}_{arm}_r{c}';b=session/f'matched_{scene.lower()}_strict_r{ref}'
                        require(physical_inputs_equal(a,b)['passed'],'Matched initial checkpoint differs')
                        comparisons.append(position_error(a/'trace/state_0001.bin',b/'trace/state_0001.bin'))
                    summary['matched'].append(dict(state=f'{scene}, step {frame}',method=arm,
                        N=statistics.median(x['N_tot'] for x in obs),P=statistics.median(x['P_tot'] for x in obs),
                        t=statistics.median(x['TimeTot'] for x in obs),
                        e_RMS_mm=max(x['rms_mm'] for x in comparisons),e_max_mm=max(x['max_mm'] for x in comparisons),raw=obs,errors=comparisons))
        else:summary['missing'].append(dict(table='matched',reason='exact checkpoint input gate failed; ablation table withheld'))
    if not perf.exists():summary['missing'].append(dict(table='performance',reason='not run'))
    write(output/'summary.json',summary)
    def csv_file(name,fields,rows):
        with (output/name).open('x',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    csv_file('phase_table.csv',['scene','method','Hess','LSolver','CCD','LineS','Misc','TimeTot','P_bar','N_bar','P_tot','S'],summary['performance'])
    graph_rows=[dict(scene=r['scene'],Base_ms=r['host']['ms_per_step'],Graph_ms=r['graph']['ms_per_step'],T_B_over_T_G=r['T_B_over_T_G'],
        Base_N=r['host']['N_bar'],Graph_N=r['graph']['N_bar'],Base_P=r['host']['P_bar'],Graph_P=r['graph']['P_bar']) for r in summary['graph']]
    csv_file('graph_table.csv',['scene','Base_ms','Graph_ms','T_B_over_T_G','Base_N','Graph_N','Base_P','Graph_P'],graph_rows)
    csv_file('checkpoint_table.csv',['state','method','N','P','t','e_RMS_mm','e_max_mm'],summary['matched'])
    def latex(name,fields,rows):
        headers={'P_bar':r'$\overline{P}$','N_bar':r'$\overline{N}$','P_tot':r'$P_{\mathrm{tot}}$',
            'e_RMS_mm':r'$e_x^{\mathrm{RMS}}$ (mm)','e_max_mm':r'$e_x^{\max}$ (mm)'}
        text=['\\begin{tabular}{'+'l'*len(fields)+'}','\\hline',
            ' & '.join(headers.get(k,k.replace('_',r'\_')) for k in fields)+r' \\','\\hline']
        previous=None
        for r in rows:
            group=r.get('scene',r.get('state'))
            if previous is not None and group!=previous:text.append('\\hline')
            previous=group
            def cell(k):
                v=r.get(k)
                if v is None:return '--'
                if k=='method':return {'stiff':r'\textsc{Base}','ours':r'\textsc{Ours}',
                    'original':'Original (Graph)','pure':'Pure TOI'}[v]
                return format(v,'.3f') if isinstance(v,float) else str(v).replace('_',r'\_')
            text.append(' & '.join(cell(k) for k in fields)+r' \\')
        text.extend(['\\hline','\\end{tabular}'])
        (output/name).write_text('\n'.join(text)+'\n',encoding='utf8')
    latex('phase_table.tex',['scene','method','Hess','LSolver','CCD','LineS','Misc','TimeTot','P_bar','N_bar','P_tot','S'],summary['performance'])
    latex('graph_table.tex',['scene','Base_ms','Graph_ms','T_B_over_T_G','Base_N','Graph_N','Base_P','Graph_P'],graph_rows)
    # Grouped header layout from the supplied Graph ablation example.
    graph_tex=[r'\begin{tabular}{lrrrrrrr}',r'\hline',
        r' & \multicolumn{3}{c}{$\overline{t}$ (ms/step)} & \multicolumn{2}{c}{$\overline{N}$} & \multicolumn{2}{c}{$\overline{P}$} \\',
        r'Scene & Base & Graph & $T_B/T_G$ & Base & Graph & Base & Graph \\',r'\hline']
    for row in graph_rows:
        graph_tex.append(row['scene']+' & '+' & '.join(format(row[k],'.3f') for k in ('Base_ms','Graph_ms','T_B_over_T_G','Base_N','Graph_N','Base_P','Graph_P'))+r' \\')
    graph_tex.extend([r'\hline',r'\end{tabular}'])
    (output/'graph_table.tex').write_text('\n'.join(graph_tex)+'\n',encoding='utf8')
    latex('checkpoint_table.tex',['state','method','N','P','t','e_RMS_mm','e_max_mm'],summary['matched'])
    print(json.dumps(dict(phase_rows=len(summary['performance']),graph_rows=len(summary['graph']),checkpoint_rows=len(summary['matched']),missing=len(summary['missing']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();tables(args.session.resolve(),args.output.resolve())
