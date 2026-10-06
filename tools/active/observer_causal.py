"""Predeclared, serial observation/binary controls; no quality reclassification."""
import argparse
import collections
import itertools
import json
import statistics
from pathlib import Path
import numpy as np
from config import ROOT,read,sha
from ipc_benchmark import metrics,write

TAG='ipc_observer_causal_20261005'
ARMS={'raw_off':('base',False),'observed_off':('base_observed',False),'observed_on':('base_observed',True)}

def prepare():
    runs=[]
    for repeat in range(1,6):
        arms=list(ARMS);shift=(repeat-1)%3;arms=arms[shift:]+arms[:shift]
        if repeat%2==0:arms=arms[::-1]
        for arm in arms:
            binary,velocity=ARMS[arm]
            runs.append({'name':f'{TAG}_{arm}_r{repeat}','arm':arm,'variant':arm,'repeat':repeat,
                'binary':binary,'config':{'scene':'cloth_fixed_bunny_l','steps':100,'dt':.01,
                    'timeout_seconds':120,'backend':'ipc','execution':'host','mas':'legacy',
                    'refit':False,'batch':False,'reuse':False,'trace_velocity':velocity,
                    'ipc_newton_tol':.01,'ipc_cumulative_tol':.01,'ipc_min_updates':6,'pcg_rho_tol':1e-4}})
    plan={'report':f'reports/active/{TAG}_batch.json','stop_on_failure':True,'runs':runs}
    write(ROOT/f'configs/active/{TAG}.json',plan)
    protocol={'schema':1,'phase':'Causal diagnosis only; no new quality acceptance envelope',
        'runs':15,'frames':100,'repeats':5,'dt':.01,'timeout_seconds':120,
        'factors':{'binary_identity':['raw_off','observed_off'],'runtime_velocity_observation':['observed_off','observed_on']},
        'hypotheses':['Binary-only differences may change numerical prefixes even with the observation branch disabled.',
            'Velocity observation may change trajectories through scheduling; it does not directly mutate GPU state.'],
        'quality_protocol_sha256':sha(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json'),
        'primary_endpoints':['whole-segment max stretch','whole-segment max p99 stretch','directions','total PCG',
            'first PCG/exit divergence','first cloth RMS >1e-8m / >1e-5m / >1e-3m'],
        'comparison':'All within-arm pairs and the 5 same-repeat between-arm pairs; binary/state identity separately validated.',
        'statistics':'Five pairs are exploratory. Exact paired sign-flip on material endpoints; no threshold revision, no equivalence certification from failure to reject.',
        'stop':'Any runtime failure stops batch; retain all runs. No new solver kernel or stopping change. Do not extend repetitions after seeing outcome.',
        'old_results':'Old material failures remain failures; new data do not retrospectively broaden the old protocol.',
        'performance':'Diagnostic only; export is outside solver timer. Not a speedup certification.'}
    write(ROOT/f'reports/active/{TAG}_protocol.json',protocol)
    print(json.dumps({'plan_runs':len(runs),'protocol':str(ROOT/f'reports/active/{TAG}_protocol.json')}))

def prefix_pair(a,b):
    fa=read(a/'output/stats.json')['frames'];fb=read(b/'output/stats.json')['frames']
    abd=read(a/'trace/metadata.json')['abd_point_num'];first={};peak=0.;rows=[]
    for f in range(101):
        x=np.fromfile(a/f'trace/state_{f:04d}.bin',dtype='<f8').reshape(-1,3)
        y=np.fromfile(b/f'trace/state_{f:04d}.bin',dtype='<f8').reshape(-1,3)
        delta=np.linalg.norm(x[abd:]-y[abd:],axis=1);rms=float(np.sqrt(np.mean(delta*delta)));peak=max(peak,rms)
        for tol in (1e-8,1e-5,1e-3):
            if rms>tol and str(tol) not in first:first[str(tol)]=f
        if f:
            ca=[n['pcg']['iterations'] for n in fa[f-1]['newton'] if 'pcg' in n]
            cb=[n['pcg']['iterations'] for n in fb[f-1]['newton'] if 'pcg' in n]
            if ca!=cb and 'pcg_signature' not in first:first['pcg_signature']=f
            if len(ca)!=len(cb) and 'directions' not in first:first['directions']=f
            al_a=[n.get('alpha') for n in fa[f-1]['newton'] if 'alpha' in n]
            al_b=[n.get('alpha') for n in fb[f-1]['newton'] if 'alpha' in n]
            if len(al_a)==len(al_b) and any(abs(x-y)>1e-12 for x,y in zip(al_a,al_b)) and 'alpha_1e-12' not in first:
                first['alpha_1e-12']=f
            geo_a=fa[f-1].get('contact_geometry',{});geo_b=fb[f-1].get('contact_geometry',{})
            counts=['native_narrow_self_pairs','native_narrow_ground_pairs']
            if any(geo_a.get(s)!=geo_b.get(s) for s in counts) and 'native_candidate_count' not in first:
                first['native_candidate_count']=f
            if fa[f-1].get('newton_exit')!=fb[f-1].get('newton_exit') and 'exit' not in first:first['exit']=f
        rows.append({'frame':f,'cloth_rms_m':rms,'cloth_max_m':float(delta.max())})
    return {'a':a.name,'b':b.name,'first_divergence':first,'max_frame_cloth_rms_m':peak,'frames':rows}

def paired_test(a,b):
    delta=np.array(b)-np.array(a);mean=float(delta.mean())
    null=[abs(float(np.mean(delta*np.array(signs)))) for signs in itertools.product([-1,1],repeat=len(delta))]
    p=sum(v>=abs(mean)-1e-15 for v in null)/len(null)
    return {'a':list(a),'b':list(b),'mean_b_minus_a':mean,'paired_differences':delta.tolist(),
        'exact_two_sided_sign_flip_p':p,'interpretation':'Exploratory with 5 pairs; nonsignificance is not equivalence.'}

def analyze():
    batch=read(ROOT/f'reports/active/{TAG}_batch.json');records=[];folders={}
    bounds=read(ROOT/'reports/active/ipc_revision_20261005_quality_protocol.json')['scenes']['fixed_bunny']['bounds']
    for t in batch['runs']:
        if t['result']['status']!='completed':raise ValueError('Incomplete causal experiment')
        folder=ROOT/'runs/active'/t['name'];r=metrics(folder);r.update(arm=t['variant'],repeat=t['repeat'])
        r['velocity_finite']=all(np.isfinite(np.fromfile(p,dtype='<f8')).all() for p in (folder/'trace').glob('velocity_*.bin'))
        if r['pcg_failures'] or not r['finite'] or not r['velocity_finite']:raise ValueError('Hard numerical failure: '+t['name'])
        r['old_protocol_material_passed']=r['finite'] and not r['pcg_failures'] and all(r[k]<=v for k,v in bounds.items())
        folders[t['variant'],t['repeat']]=folder;records.append(r)
    attributes=['state_0000.bin','topology.bin','masses.bin','boundary_types.bin','body_ids.bin']
    identity={s:len({sha(p/'trace'/s) for p in folders.values()})==1 for s in attributes}
    summaries={}
    for arm in ARMS:
        rs=sorted([r for r in records if r['arm']==arm],key=lambda r:r['repeat'])
        summaries[arm]={'bounds':bounds,'material_passed':sum(r['old_protocol_material_passed'] for r in rs),
            'max_stretch':[r['max_stretch'] for r in rs],'p99_stretch':[r['p99_stretch'] for r in rs],
            'directions':[r['directions'] for r in rs],'pcg':[r['pcg'] for r in rs],
            'velocity_frames':[r['velocity_frames'] for r in rs],
            'seconds_median':statistics.median(r['seconds'] for r in rs)}
    pairs={};effects={}
    for key,(aa,bb) in [('binary_identity',('raw_off','observed_off')),('velocity_observation',('observed_off','observed_on'))]:
        pairs[key]=[prefix_pair(folders[aa,i],folders[bb,i]) for i in range(1,6)]
        effects[key]={s:paired_test(summaries[aa][s],summaries[bb][s]) for s in ['max_stretch','p99_stretch','pcg','directions']}
    for arm in ARMS:
        pairs['within_'+arm]=[prefix_pair(folders[arm,a],folders[arm,b]) for a,b in itertools.combinations(range(1,6),2)]
    report={'schema':1,'protocol_sha256':sha(ROOT/f'reports/active/{TAG}_protocol.json'),
        'identity':identity,'summary':summaries,'effects':effects,'pairs':pairs,
        'runs':records,'quality_certified':False,'performance_certified':False,'old_quality_bounds_unchanged':True,
        'statistical_limits':'Sign-flip enumeration assumes symmetric/exchangeable paired differences. Fixed balanced ordering is not a randomized trial; p-values here are descriptive, not formal causal certification.'}
    write(ROOT/f'reports/active/{TAG}_analysis.json',report)
    print(json.dumps({'identity':identity,'summary':summaries,'effects':effects}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','analyze']);a=p.parse_args()
    {'prepare':prepare,'analyze':analyze}[a.action]()
