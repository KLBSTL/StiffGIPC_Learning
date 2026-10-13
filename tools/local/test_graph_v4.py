"""Bounded retest of the exact hash-recovered early Graph source."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
from config import read,expand,digest,environment
from linux_runner import require,write_new
from local_identity import record,verify,text,norm
import windows_runner
from pool_metrics import metrics,export_evidence,state_comparison
from quality_analysis import input_comparison
from compare_graph_history import identities,task,power

SOURCE=ROOT/'build/graph_v4_recovered_source'
BUILD=ROOT/'build/graph_v4_rebuild_20261011'
SCENES=('cloth_hang_l','cloth_fixed_bunny_l')
ARMS=('base','early_host','early_graph','current_host','current_graph')


def seal():
    recovery=read(SOURCE/'RECOVERY.json')
    require(recovery['recovered']==recovery['expected']==278 and not recovery['missing'],'Exact recovery required')
    manifest=read(recovery['historical_manifest']);sources=[]
    for r in manifest['files']:
        if not r['path'].startswith('sources/stiff_fused/'):continue
        p=SOURCE/r['path'].removeprefix('sources/stiff_fused/');actual=record(p)
        require((actual['bytes'],actual['sha256'])==(r['bytes'],r['sha256']),'Historical bytes differ')
        sources.append(actual)
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    project=ET.parse(BUILD/'gipc.vcxproj')
    units=[]
    for tag in ('CudaCompile','ClCompile'):
        for node in project.findall('.//m:'+tag,ns):
            if node.get('Include'):
                units.append((BUILD/node.get('Include')).resolve())
    expected={p.resolve() for p in (SOURCE/'StiffGIPC').rglob('*') if p.suffix.lower() in ('.cu','.cpp')}
    require(set(units)==expected and len(units)==len(expected),'Native source coverage differs')
    logpath=ROOT/'reports/local_graph_v4_build_20261011.log';log=text(logpath)
    commands=[line.strip() for line in log.splitlines() if 'nvcc.exe' in line.lower()
              and any(flag in line.lower() for flag in (' -c ', ' --compile ', ' -x cu '))
              and ' -dlink ' not in line.lower()]
    require(all(any(norm(p) in norm(line) for line in commands) for p in units if p.suffix.lower()=='.cu'),'Native CUDA compile command missing')
    tlogs=list((BUILD/'gipc.dir/Release/gipc.tlog').glob('*.tlog'))
    require(tlogs,'Tracking evidence absent')
    linkpath=BUILD/'gipc.dir/Release/gipc.tlog/link.read.1.tlog';link=text(linkpath)
    objects=[record(p) for p in BUILD.rglob('*.obj') if 'gipc.dir' in p.parts and norm(p) in norm(link)]
    require(len(objects)>=len(units),'Native linked object coverage incomplete')
    exe=record(BUILD/'Release/gipc.exe')
    return dict(kind='early_v4_rebuilt',exe=exe,dlls=[record(p) for p in sorted((BUILD/'Release').glob('*.dll'))],
        sources=sources,source_root=str(SOURCE),source_digest=digest(sources),
        objects=objects,translation_units=len(units),cuda_commands=commands,
        build_evidence=[record(p) for p in [BUILD/'gipc.vcxproj',BUILD/'CMakeCache.txt',logpath,
            ROOT/'reports/local_graph_v4_configure_20261011.log',SOURCE/'RECOVERY.json',Path(recovery['historical_manifest']),*tlogs]],
        capabilities=dict(resolved_config=False,actual_velocity=False),
        scope='Exact 278 recorded source/input file hashes; newly rebuilt on current CUDA/MSVC. Original historical executable hash is not reproduced.')


def legacy_validate(out):
    requested=read(out/'requested.json');c=requested['expanded_config'];r=read(out/'result.json')
    frames=read(out/'output/stats.json')['frames']
    pcgs=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    execution=c['execution']
    checks=dict(complete=r['status']=='completed' and len(frames)==c['steps'],
        no_pcg_cap=not any(p.get('iteration_limit') or p.get('breakdown') for p in pcgs),
        actual_graph=bool(pcgs) and all(p.get('execution','host')==execution for p in pcgs),
        finite_state=export_evidence(out,len(frames),'state')['passed'])
    return dict(passed=all(checks.values()),checks=checks,
        scope='Old frontend has no resolved-config/velocity telemetry; execution observed in every PCG record. Scalars bound by exact source plus explicit environment, not runtime-resolved fields.')


def summarize(out,result):
    m=metrics(out);frames=read(out/'output/stats.json')['frames']
    pcgs=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    require(m['finite'] and not m['pcg_failures'] and export_evidence(out,len(frames),'state')['passed'],'Invalid state/PCG/export')
    require(not any(f.get('newton_exit')=='iteration_limit' for f in frames),'Nonlinear cap')
    return dict(solver_seconds=result['solver_seconds'],frames=len(frames),
        directions=m['directions'],pcg=m['pcg'],pcg_per_direction=m['pcg']/m['directions'],
        phase_ms=m['phase_ms'],max_stretch=m['max_stretch'],p99_stretch=m['p99_stretch'],
        fixed_drift_m=m['fixed_drift_m'],
        graph_captures=sum(p.get('execution')=='conditional_graph' and not p.get('graph_cache_hit',False) for p in pcgs),
        graph_capture_ms=sum(p.get('graph_capture_instantiate_host_ms',0) for p in pcgs),
        physical_quality_certified=False)


def run(name,stage):
    session=ROOT/'runs'/name;session.mkdir(parents=True,exist_ok=True)
    require(not (session/(stage+'_ledger.json')).exists(),'Stage exists')
    ids=identities();ids['early']=seal()
    packet=dict(programs=ids,controller=record(__file__),power=power(),frames=100,repeats=3,arms=ARMS)
    identitypath=session/'identities.json'
    if identitypath.exists():require(read(identitypath)==json.loads(json.dumps(packet)),'Identity/power/controller changed')
    else:write_new(identitypath,packet)
    if stage=='performance':require(read(session/'smoke_ledger.json')['complete'],'Smoke checks required')
    tasks=[]
    for repeat in range(1,2 if stage=='smoke' else 4):
        order=ARMS[repeat-1:]+ARMS[:repeat-1]
        for scene in SCENES if repeat%2 else tuple(reversed(SCENES)):
            for arm in order:
                t=task(scene,'frozen_stiff' if arm=='base' else 'current_graph' if arm.endswith('graph') else 'current_host',repeat,2 if stage=='smoke' else 100,stage)
                t.update(name=f'{stage}_{scene}_{arm}_r{repeat}',arm=arm,
                    identity_key='base' if arm=='base' else 'early' if arm.startswith('early') else 'active')
                tasks.append(t)
    write_new(session/(stage+'_tasks.json'),dict(tasks=tasks))
    old_env,old_validate=windows_runner.environment,windows_runner.validate
    rows=[]
    try:
        with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
            for t in tasks:
                identity=ids[t['identity_key']]
                def env(c,out):
                    value=environment(c,out);value['GIPC_IPC_PCG_FORCING']='off';return value
                windows_runner.environment=env
                windows_runner.validate=legacy_validate if t['identity_key']=='early' else old_validate
                result=windows_runner.execute(session,t,identity,0)
                row=dict(task=t,result=result);rows.append(row)
                if result['status']!='completed':break
                try:row['summary']=summarize(session/t['name'],result)
                except Exception as error:row['analysis_error']=str(error);break
                with (session/(stage+'_journal.jsonl')).open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(dict(name=t['name'],**row['summary'])),flush=True)
    finally:windows_runner.environment,windows_runner.validate=old_env,old_validate
    final=dict(runs=rows,skipped=tasks[len(rows):],complete=len(rows)==len(tasks) and all('summary' in r for r in rows),power_after=power())
    write_new(session/(stage+'_ledger.json'),final)
    require(final['power_after']==packet['power'],'Power changed')
    verify(ids['early']['sources']+ids['early']['build_evidence']+ids['early']['objects']+[ids['early']['exe']]+ids['early']['dlls'])
    print(json.dumps(dict(stage=stage,complete=final['complete'],runs=len(rows))),flush=True)
    return 0 if final['complete'] else 2


def analyze(name):
    session=ROOT/'runs'/name;ledger=read(session/'performance_ledger.json');require(ledger['complete'],'Incomplete matrix')
    rows=ledger['runs'];pairs=[];table=[]
    for scene in SCENES:
        for repeat in range(1,4):
            by={r['task']['arm']:r for r in rows if r['task']['scene']==scene and r['task']['repeat']==repeat}
            for reference,candidate in [('base',a) for a in ARMS[1:]]+[
                ('early_host','early_graph'),('current_host','current_graph'),('early_graph','current_graph'),('early_host','current_host')]:
                a,b=by[reference],by[candidate]
                initial=input_comparison(session/a['task']['name'],session/b['task']['name'])
                require(initial['passed'],'Different initial inputs')
                pairs.append(dict(scene=scene,repeat=repeat,reference=reference,candidate=candidate,
                    speedup=a['summary']['solver_seconds']/b['summary']['solver_seconds'],initial=initial,
                    trajectory=state_comparison(session/a['task']['name'],session/b['task']['name'])))
        for arm in ARMS:
            rs=[r for r in rows if r['task']['scene']==scene and r['task']['arm']==arm]
            values={k:[r['summary'][k] for r in rs] for k in ['solver_seconds','directions','pcg','pcg_per_direction','max_stretch','p99_stretch','graph_captures','graph_capture_ms']}
            speeds=[p['speedup'] for p in pairs if p['scene']==scene and p['reference']=='base' and p['candidate']==arm]
            table.append(dict(scene=scene,arm=arm,medians={k:statistics.median(v) for k,v in values.items()},raw=values,
                              speedup_median=statistics.median(speeds) if speeds else None))
    write_new(session/'analysis.json',dict(rows=table,pairs=pairs,physical_quality_certified=False,scope='Exact source restored; current-compiler rebuild, 100 frames x 3'))
    print(json.dumps(table,indent=2))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('smoke','performance','analyze'))
    parser.add_argument('--session',default='local_graph_v4_20261011')
    args=parser.parse_args()
    raise SystemExit(analyze(args.session) if args.stage=='analyze' else run(args.session,args.stage))
