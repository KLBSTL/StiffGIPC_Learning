"""Recompute native observations and paper-style tables for the local combination suite."""
import argparse
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import statistics
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/paper_suite'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from run_combination_suite import plan,ARMS,SCENES,CHECKPOINTS
from analyze import observation,physical_inputs_equal,position_error,checkpoint_gate
from config import read
from linux_runner import require,write_new,sha
from local_identity import verify
from pool_metrics import state_comparison

FIELDS=('Hess','LSolver','CCD','LineS','Misc','TimeTot','P_bar','N_bar','P_tot','N_tot','ms_per_step')
PHASE_FIELDS=('scene','method','Hess','LSolver','CCD','LineS','Misc','TimeTot','P_bar','N_bar','P_tot','S')
GRAPH_FIELDS=('scene','Base_ms','Graph_ms','T_B_over_T_G','Base_N','Graph_N','Base_P','Graph_P')
CHECK_FIELDS=('state','method','N','P','t','e_RMS_mm','e_max_mm')


def csv_file(output,name,fields,rows):
    with (output/(name+'.csv')).open('x',encoding='utf8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def table_tex(output,name,headers,rows,group_index=0):
    text=['\\begin{tabular}{'+'ll'+'r'*(len(headers)-2)+'}',r'\hline',' & '.join(headers)+r' \\',r'\hline']
    for i,row in enumerate(rows):
        if i and row[group_index]!=rows[i-1][group_index]:text.append(r'\hline')
        text.append(' & '.join(str(v).replace('_',r'\_') for v in row)+r' \\')
    text.extend([r'\hline',r'\end{tabular}'])
    (output/(name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf8')


def ground_observation(folder,steps):
    meta=read(folder/'trace/metadata.json');n=np.array(meta['ground_normal'],float)
    boundary=np.fromfile(folder/'trace/boundary_types.bin',dtype='<i4')
    values=[]
    for frame in range(steps+1):
        x=np.fromfile(folder/f'trace/state_{frame:04d}.bin',dtype='<f8').reshape(-1,3)
        require(len(x)==len(boundary) and np.isfinite(x).all(),'Invalid finite state')
        gap=x[boundary==0]@n-float(meta['ground_offset'])
        if gap.size:values.append(float(gap.min()))
    minimum=min(values) if values else None
    return dict(minimum_dynamic_vertex_ground_gap_m=minimum,
        sampled_ground_penetration_m=max(0,-minimum) if minimum is not None else None,
        scope='Exported vertex-to-ground distance at frame boundaries only; no self/obstacle collision certification')


def make_tables(output,groups,graphs,matched):
    spec=importlib.util.spec_from_file_location('previous_paper_table_plot',ROOT/'runs/autodl_20261011/format_tables.py')
    plotting=importlib.util.module_from_spec(spec);spec.loader.exec_module(plotting)
    headers=['Scene','Method','Hess','LSolver','CCD','LineS','Misc','TimeTot',r'$\overline{P}$',r'$\overline{N}$',r'$P_{tot}$','S']
    widths=[.85,1,.75,.85,.75,.75,.75,.9,.8,.8,.9,.75]
    for name,arms in [('phase_table_current',('stiff','ours')),('phase_table_frozen',('stiff','old')),
                      ('phase_table_both',('stiff','old','ours'))]:
        selected=[r for r in groups if r['method'] in arms]
        csv_file(output,name,PHASE_FIELDS,selected)
        labels={'stiff':'Base','old':'Old','ours':'Current'}
        rows=[[r['scene'],labels[r['method']],*[f'{r[k]:.3f}' for k in FIELDS[:6]],
               f'{r["P_bar"]:.2f}',f'{r["N_bar"]:.2f}',str(int(r['P_tot'])),f'{r["S"]:.3f}'] for r in selected]
        caption='Local RTX 3070 Laptop; dt=.01, rho=1e-4; 100 frames x 3 interleaved rounds. Times: seconds/100 frames.\n'+\
            'Base: frozen Stiff. Old/Current: complete execution combination, legacy cumulative .01. S: median paired Base/candidate.'
        plotting.plot(output,name,headers,rows,widths,group_key=0,caption=caption)
        table_tex(output,name,headers,rows)
    for version in ('old','current'):
        selected=[r for r in graphs if r['version']==version];name='graph_table_'+version
        csv_file(output,name,GRAPH_FIELDS,selected)
        rows=[[r['scene'],*[f'{r[k]:.1f}' for k in ('Base_ms','Graph_ms')],f'{r["T_B_over_T_G"]:.3f}',
               *[f'{r[k]:.2f}' for k in ('Base_N','Graph_N','Base_P','Graph_P')]] for r in selected]
        plotting.plot(output,name,['Scene','Host','Graph',r'$T_B/T_G$','Host','Graph','Host','Graph'],rows,[1.2,1,1,1,1,1,1,1],
            groups=[(1,4,r'$\overline{t}$ (ms/step)'),(4,6,r'$\overline{N}$'),(6,8,r'$\overline{P}$')],
            caption=f'{version.title()} binary, complete combination: only execution differs. dt=.01, rho=1e-4; 100 frames x 3 rounds.\n'+
                    'Host is the same combination without Graph, distinct from frozen Stiff in the phase table.')
        tex=[r'\begin{tabular}{lrrrrrrr}',r'\hline',
             r' & \multicolumn{3}{c}{$\overline{t}$ (ms/step)} & \multicolumn{2}{c}{$\overline{N}$} & \multicolumn{2}{c}{$\overline{P}$} \\',
             r'Scene & Host & Graph & $T_B/T_G$ & Host & Graph & Host & Graph \\',r'\hline']
        tex+=[' & '.join(row)+r' \\' for row in rows];tex+=[r'\hline',r'\end{tabular}']
        (output/(name+'.tex')).write_text('\n'.join(tex)+'\n',encoding='utf8')
    csv_file(output,'checkpoint_table',CHECK_FIELDS,matched)
    labels={'original':'Movement only (Graph)','pure':'Cumulative TOI .01','ours':'Residual gate (beta=.01)'}
    rows=[[r['state'],labels[r['method']],f'{r["N"]:g}',f'{r["P"]:.1f}',f'{r["t"]:.4f}',
           f'{r["e_RMS_mm"]:.6f}',f'{r["e_max_mm"]:.4f}'] for r in matched]
    headers=['State','Outer rule','N','P','t (s)',r'$e_x^{RMS}$ (mm)',r'$e_x^{max}$ (mm)']
    plotting.plot(output,'checkpoint_table',headers,rows,[1.65,2.5,.5,.8,.8,1.6,1.6],group_key=0,
        caption='Same current binary and cold checkpoint; complete combination, dt=.01; candidate rho=1e-4, movement threshold .01.\n'+
                'Errors: worst 3 x 3 pairings against movement-only rho=1e-6 references. Residual gate .03, compensated budget .01.\n'+
                'Exact input gate and hot/cold continuity reported separately; numerical reference is not an exact solution.')
    table_tex(output,'checkpoint_table',headers,rows)


def analyze(session,output):
    require(not output.exists(),'Output exists; do not replace results');output.mkdir(parents=True)
    ids=read(session/'identities.json');require(ids['plan']==json.loads(json.dumps(plan())),'Recorded plan changed')
    ledger_path=session/'performance_delivery_ledger.json'
    packet=read(ledger_path if ledger_path.exists() else session/'performance_ledger.json')
    selected_scenes=packet.get('selected_scenes',list(SCENES))
    excluded_scenes=packet.get('excluded_scenes',[])
    require(packet['complete'] and not packet['skipped'],'Incomplete selected matrix')
    require(len(packet['runs'])==15*len(selected_scenes),'Missing formal runs')
    expected={t['name'] for t in plan()['stages']['performance'] if t['scene_key'] in selected_scenes}
    require({r['task']['name'] for r in packet['runs']}==expected,'Selected matrix task coverage differs')
    if excluded_scenes:
        require(excluded_scenes==['Stack'] and packet['scope_reduction']['authorized_by_user']=='Stack暂缓，先交付六个场景','Unapproved scope reduction')
    lookup={};raw=[]
    for r in packet['runs']:
        t=r['task'];folder=session/t['name'];native=read(folder/'result.json')
        require(native['status']=='completed' and native['recorded_frames']==100,'Native incomplete')
        value=observation(folder,t,native);require(value==r['observation'],'Raw observations changed')
        value['ground']=ground_observation(folder,100)
        value['linear_ms_per_pcg']=1000*value['LSolver']/value['P_tot']
        lookup[t['scene_key'],t['arm'],t['repeat']]=value
        raw.append(dict(scene=t['scene_key'],method=t['arm'],repeat=t['repeat'],name=t['name'],**value))
    groups=[];pairs=[];graphs=[]
    for scene in selected_scenes:
        for repeat in (1,2,3):
            for reference,candidate in [('stiff','old'),('stiff','ours'),('old_host','old'),('host','ours'),('old','ours')]:
                a=session/f'performance_{scene.lower()}_{reference}_r{repeat}';b=session/f'performance_{scene.lower()}_{candidate}_r{repeat}'
                inputs=physical_inputs_equal(a,b);require(inputs['passed'],'Different physical inputs')
                lhs=lookup[scene,reference,repeat];rhs=lookup[scene,candidate,repeat]
                pairs.append(dict(scene=scene,repeat=repeat,reference=reference,candidate=candidate,input=inputs,
                    speedup=lhs['TimeTot']/rhs['TimeTot'],pcg_ratio=rhs['P_tot']/lhs['P_tot'],direction_ratio=rhs['N_tot']/lhs['N_tot'],
                    linear_time_ratio=rhs['LSolver']/lhs['LSolver'],linear_per_pcg_ratio=rhs['linear_ms_per_pcg']/lhs['linear_ms_per_pcg'],
                    trajectory=state_comparison(a,b)))
        for arm in ARMS:
            values=[lookup[scene,arm,r] for r in (1,2,3)]
            groups.append(dict(scene=scene,method=arm,**{k:statistics.median(v[k] for v in values) for k in FIELDS},
                S=1 if arm=='stiff' else statistics.median(lookup[scene,'stiff',r]['TimeTot']/lookup[scene,arm,r]['TimeTot'] for r in (1,2,3)),raw=values))
        for version,host,graph in [('old','old_host','old'),('current','host','ours')]:
            graphs.append(dict(scene=scene,version=version,
                **{prefix+'_'+k:statistics.median(lookup[scene,arm,r][source] for r in (1,2,3))
                   for prefix,arm in [('Base',host),('Graph',graph)] for k,source in [('ms','ms_per_step'),('N','N_bar'),('P','P_bar')]},
                T_B_over_T_G=statistics.median(lookup[scene,host,r]['TimeTot']/lookup[scene,graph,r]['TimeTot'] for r in (1,2,3))))
    matched_packet=read(session/'matched_ledger.json');require(matched_packet['complete'],'Incomplete matched matrix')
    gate=checkpoint_gate(session,matched_packet);require(gate['passed'],'Checkpoint input gate failed')
    require(gate==read(session/'checkpoint_input_gate.json'),'Checkpoint gate changed')
    matched=[]
    for scene,frame in CHECKPOINTS.items():
        for arm in ('original','pure','ours'):
            selected=[r for r in matched_packet['runs'] if r['task']['scene_key']==scene and r['task']['arm']==arm]
            require(len(selected)==3,'Three matched runs required')
            values=[observation(session/r['task']['name'],r['task'],read(session/r['task']['name']/'result.json')) for r in selected]
            errors=[]
            for c,ref in itertools.product((1,2,3),(1,2,3)):
                a=session/f'matched_{scene.lower()}_{arm}_r{c}';b=session/f'matched_{scene.lower()}_strict_r{ref}'
                require(physical_inputs_equal(a,b)['passed'],'Matched initial inputs differ')
                errors.append(position_error(a/'trace/state_0001.bin',b/'trace/state_0001.bin'))
            matched.append(dict(state=f'{scene}, step {frame}',method=arm,
                N=statistics.median(v['N_tot'] for v in values),P=statistics.median(v['P_tot'] for v in values),
                t=statistics.median(v['TimeTot'] for v in values),e_RMS_mm=max(e['rms_mm'] for e in errors),e_max_mm=max(e['max_mm'] for e in errors),raw=values,errors=errors))
    total={}
    for arm in ('old','ours'):
        ratios=[sum(lookup[s,'stiff',r]['TimeTot'] for s in selected_scenes)/sum(lookup[s,arm,r]['TimeTot'] for s in selected_scenes) for r in (1,2,3)]
        total[arm]=dict(median=statistics.median(ratios),raw=ratios)
    identity_rows=[]
    for name,program in ids['programs'].items():
        files=program['sources']+program['build_evidence']+[program['exe']]+program['dlls']+program.get('objects',[])
        verify(files);identity_rows.append(dict(program=name,verified_files=len(files),exe_sha256=program['exe']['sha256']))
    verify([ids['controller']]);require(packet['power_after']==ids['power'],'Power changed')
    summary=dict(complete=True,plan=ids['plan'],selected_scenes=selected_scenes,excluded_scenes=excluded_scenes,
        formal_runs=len(packet['runs']),formal_frames=100*len(packet['runs']),groups=groups,pairs=pairs,
        graph=graphs,matched=matched,checkpoint_gate=gate,aggregate_speedups=total,identity_recheck=identity_rows,
        checkpoint_origin=dict(actual_origin='Native checkpoint restore',
            generic_windows_request_from_zero_field_is_not_authoritative=True,
            evidence='GIPC_CHECKPOINT_LOAD, native checkpoint_restore.json, exact initial position/velocity/buffer/context gate'),
        source_controller=record_of_script(),performance_certified=False,physical_quality_certified=False)
    write_new(output/'summary.json',summary)
    csv_file(output,'all_methods',('scene','method',*FIELDS,'S'),groups)
    csv_file(output,'raw_observations',('scene','method','repeat','name',*FIELDS,'linear_ms_per_pcg'),raw)
    csv_file(output,'paired_work_cost',('scene','repeat','reference','candidate','speedup','pcg_ratio','direction_ratio','linear_time_ratio','linear_per_pcg_ratio'),pairs)
    quality=[dict(scene=r['scene'],method=r['method'],repeat=r['repeat'],**r['quality'],**r['ground']) for r in raw]
    csv_file(output,'quality_observations',tuple(quality[0]),quality)
    make_tables(output,groups,graphs,matched)
    print(json.dumps(dict(complete=True,formal_runs=len(packet['runs']),formal_frames=100*len(packet['runs']),input_pairs=len(pairs),checkpoint_gate=gate['passed'],aggregate=total)))


def record_of_script():return dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.session.resolve(),a.output.resolve())
