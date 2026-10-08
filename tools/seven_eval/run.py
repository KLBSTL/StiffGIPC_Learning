"""One explicitly requested AutoDL task per invocation; never retries or builds."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/full_eval'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
import analysis as metrics
import build_identity as builds
import linux_runner as linux
from fixed_quality import execute_base
from config import read,expand,digest
from linux_runner import require,sha,child,inventory,verify_files,write_new

SCENES=('cloth_hang_l','cloth_hang_m','cloth_sphere7_l','cloth_sphere7_m',
        'cloth_fixed_bunny_l','cloth_fixed_bunny_m','bunny_cloth_bunny_l')
ARMS=('base','graph','all')
REPORT='reports/autodl_seven_20261008'
SESSION='runs/autodl_seven_20261008'

def tasks():
    rows=[]
    for i,scene in enumerate(SCENES):
        order=ARMS[i%3:]+ARMS[:i%3]
        for arm in order:
            c={'scene':scene,'preset':'combined' if arm=='all' else 'base',
               'backend':'ipc','execution':'host' if arm=='base' else 'conditional_graph',
               'mas':'legacy','steps':120,'dt':.01,'timeout_seconds':180,
               'ipc_termination':'legacy','ipc_newton_tol':.01,'ipc_cumulative_tol':.01,
               'ipc_min_updates':6,'pcg_rho_tol':1e-4,'pcg_graph_chunk':1,
               'discrete_bvh_refit':arm=='all','discrete_bvh_rebuild_interval':8,
               'contact_pool':False,'trace_velocity':arm!='base','diagnostics':[],'profile':'none'}
            rows.append({'index':len(rows)+1,'name':f'{len(rows)+1:02d}_{scene}_{arm}',
                         'scene_key':scene,'arm':arm,'variant':arm,'binary':'base' if arm=='base' else 'active',
                         'phase':'single_diagnostic','repeat':1,'config':c,'expanded_config_sha256':digest(expand(c))})
    return rows

def control_files():
    paths=[]
    for name in ('seven_eval','bench','full_eval','diagnostic'):
        paths.extend(p for p in (ROOT/'tools'/name).rglob('*.py') if '__pycache__' not in p.parts)
    return inventory(ROOT,paths)

def input_references():
    rows=[]
    for scene in SCENES:
        relative='benchmark_scenes/'+scene+'.json'
        spec=read(ROOT/'Assets'/relative);names=[relative,'scene/parameterSetting.txt']
        require(spec['case_id']==scene,'Scene ID differs')
        for obj in spec['objects']:
            mesh=Path(obj['stiff_mesh']);names.append(mesh.as_posix())
            require(sha(ROOT/'Assets'/mesh)==obj['stiff_mesh_sha256'],'Scene mesh hash differs')
            if obj['dimension']==2 or obj['body_type']=='FEM':
                names += ['sorted_mesh/'+mesh.stem+'_sorted.16'+mesh.suffix,
                          'sorted_mesh/'+mesh.stem+'_sorted.16.part']
        pairs=[]
        for name in sorted(set(names)):
            a=ROOT/'Assets'/name;b=ROOT/'baseline/Assets'/name
            require(a.is_file() and b.is_file() and sha(a)==sha(b),'Shared input/cache differs: '+name)
            pairs.append({'path':name,'bytes':a.stat().st_size,'sha256':sha(a)})
        rows.append({'scene':scene,'files':pairs})
    return rows

def seal():
    path=ROOT/REPORT/'SEAL.json';require(not path.exists(),'Seal already exists')
    packet_path=ROOT/REPORT/'BUILD_IDENTITY.json'
    packet=builds.verify_identity(ROOT,packet_path)
    manifests={}
    for kind in ('active','base'):
        observed=packet['builds'][kind]
        sources=linux.source_inventory(ROOT) if kind=='active' else inventory(ROOT,[ROOT/r['path'] for r in packet['sources'] if r['path'].startswith('baseline/')])
        evidence=inventory(ROOT,[ROOT/r['path'] for r in observed['build_evidence']]+[ROOT/r['path'] for r in observed['build_logs'].values()])
        manifests[kind+'_manifest']={'sources':sources,'source_digest':digest(sources),
            'exe_path':observed['exe']['path'],'exe_sha256':observed['exe']['sha256'],
            'build_evidence':evidence,'compile_source_count':observed['counts']['tu']}
    value={'schema':'seven_eval.v1','tasks':tasks(),'frames':120,'maximum_runs':21,
           'automatic_retry':False,'automatic_advance':True,'timeout_seconds':180,'controller_files':control_files(),
           'build_identity_sha256':sha(packet_path),'selected_inputs':input_references(),**manifests,
           'quality_certified':False,'performance_certified':False,'solver_commit':'3d8c15e'}
    write_new(path,value)
    session=ROOT/SESSION;require(not session.exists(),'Session already exists')
    session.mkdir(parents=True)
    write_new(session/'plan.json',{'tasks':tasks(),'seal_sha256':sha(path)})
    return {'status':'sealed','tasks':21,'frames':120}

def verify_seal():
    value=read(ROOT/REPORT/'SEAL.json')
    require(value['schema']=='seven_eval.v1' and value['tasks']==tasks(),'Declared plan changed')
    require(value['controller_files']==control_files(),'Controller changed')
    require(value['selected_inputs']==input_references(),'Selected inputs changed')
    builds.verify_identity(ROOT,ROOT/REPORT/'BUILD_IDENTITY.json',value['build_identity_sha256'])
    for kind in ('active','base'):
        m=value[kind+'_manifest'];verify_files(ROOT,m['sources']);verify_files(ROOT,m['build_evidence'])
        require(sha(ROOT/m['exe_path'])==m['exe_sha256'],'Program changed')
    return value

def records():
    rows=[]
    for index,path in enumerate(sorted((ROOT/REPORT/'runs').glob('*.json')),1):
        row=read(path)
        require(path.name==f'{index:02d}.json' and row['task']==tasks()[index-1],'Run order/task differs')
        require(row['seal_sha256']==sha(ROOT/REPORT/'SEAL.json'),'Run seal differs')
        run=ROOT/SESSION/row['task']['name']
        require(sha(run/'evidence.json')==row['evidence_sha256'],'Evidence receipt changed')
        verify_files(run,read(run/'evidence.json')['files'])
        rows.append(row)
    return rows

def ratios(rows,inputs_equal):
    arms={r['arm']:r for r in rows}
    if not inputs_equal or set(arms)!=set(ARMS) or not all(r['hard_checks_passed'] for r in rows): return None
    seconds={a:arms[a]['timing']['solver_seconds'] for a in ARMS}
    if not all(metrics.finite(s) and s>0 for s in seconds.values()): return None
    return {'base':1.0,'graph':seconds['base']/seconds['graph'],
            'all':seconds['base']/seconds['all'],'other_components':seconds['graph']/seconds['all']}

def summarize(rows):
    scenes=[]
    for scene in SCENES:
        group=[r['analysis'] for r in rows if r['task']['scene_key']==scene]
        inputs={r['arm']:r.get('input_identity') for r in group}
        proof=[]
        if inputs.get('base'):
            for arm in ('graph','all'):
                if inputs.get(arm):proof.append(metrics.compare_inputs(inputs['base'],inputs[arm]))
        equal=len(proof)==2 and all(p['passed'] for p in proof)
        scenes.append({'scene':scene,'runs':group,'input_comparisons':proof,'ratios':ratios(group,equal),
                       'quality_certified':False,'performance_certified':False})
    return {'recorded_tasks':len(rows),'planned_tasks':21,'scenes':scenes,
            'quality_certified':False,'performance_certified':False,
            'scope':'Single sample per scene/arm. CPU synchronized solver envelope. No statistical or same-quality certification.'}

def run(index):
    require(type(index)==int and 1<=index<=21,'Index must be 1..21')
    value=verify_seal();rows=records();require(index==len(rows)+1,'Only the next declared task may run; no retry')
    task=tasks()[index-1];session=ROOT/SESSION
    with linux.gpu_lock(ROOT):
        result=(execute_base if task['binary']=='base' else linux.execute)(ROOT,session,task,value[task['binary']+'_manifest'],0)
    analysis=metrics.analyze_run(session,task,value)
    row={'task':task,'result':result,'analysis':analysis,
         'evidence_sha256':sha(session/task['name']/'evidence.json'),'seal_sha256':sha(ROOT/REPORT/'SEAL.json')}
    write_new(ROOT/REPORT/'runs'/f'{index:02d}.json',row)
    summary=summarize(rows+[row]);write_new(ROOT/REPORT/'summaries'/f'{index:02d}.json',summary)
    print(json.dumps({'index':index,'run':task['name'],'status':result['status'],
                      'hard_checks_passed':analysis['hard_checks_passed'],'failures':analysis['failures'],
                      'solver_seconds':result.get('solver_seconds')}),flush=True)
    return row

def report():
    verify_seal();rows=records();value=summarize(rows)
    require(len(rows)==21,'All 21 declared outcomes are required before the final report')
    for scene in value['scenes']:
        if scene['ratios'] is None: continue
        by_arm={r['arm']:ROOT/SESSION/r['name'] for r in scene['runs']}
        scene['state_comparisons']={left+'/'+right:metrics.state_comparison(by_arm[left],by_arm[right])
            for left,right in [('base','graph'),('base','all'),('graph','all')]}
        base=next(r['material'] for r in scene['runs'] if r['arm']=='base')
        scene['signed_material_differences']={r['arm']:{key:r['material'][key]-base[key]
            if r['material'][key] is not None and base[key] is not None else None for key in base}
            for r in scene['runs'] if r['arm']!='base'}
    write_new(ROOT/REPORT/'RESULTS.json',value)
    with (ROOT/REPORT/'TIMINGS.csv').open('x',newline='') as f:
        names=['scene','arm','status','hard_checks_passed','solver_seconds','core_event_seconds','wall_seconds','pcg_iterations','linear_directions','assembly_seconds','linear_seconds','ccd_seconds','line_search_seconds','state_update_seconds','exit_assembly_seconds']
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for row in rows:
            a=row['analysis'];t=a.get('timing',{});ph=t.get('phase_ms',{});work=a.get('work',{})
            record={'scene':row['task']['scene_key'],'arm':row['task']['arm'],'status':row['result']['status'],
                    'hard_checks_passed':a['hard_checks_passed'],'solver_seconds':t.get('solver_seconds'),
                    'core_event_seconds':t.get('core_cuda_event_seconds'),'wall_seconds':t.get('process_wall_seconds'),
                    'pcg_iterations':work.get('pcg_iterations'),'linear_directions':work.get('linear_directions')}
            for dest,source in [('assembly','assembly'),('linear','pcg'),('ccd','ccd'),('line_search','line_search'),('state_update','state_update')]:
                record[dest+'_seconds']=ph[source]/1000 if ph.get(source) is not None else None
            record['exit_assembly_seconds']=t['exit_assembly_ms']/1000 if t.get('exit_assembly_ms') is not None else None
            writer.writerow(record)
    lines=['# AutoDL 七场景 120 帧单次测试','',
           '原版 Stiff 为分母；all = conditional Graph + FullCCD refit + 批量能量 + 能量复用 + 普通 BVH refit（周期 8）。',
           'rho=1e-4，dt=.01，legacy MAS/停止规则，完整 CCD。每项一次；无同质量或统计性能认证。','',
           '| 场景 | base 秒（1×） | Graph 秒 | Graph/base 加速 | all 秒 | all/base 加速 | 其他组件增量 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for scene in value['scenes']:
        arms={r['arm']:r for r in scene['runs']};r=scene['ratios']
        times=[arms.get(a,{}).get('timing',{}).get('solver_seconds') for a in ARMS]
        fmt=lambda v:f'{v:.4f}' if v is not None else '失败/未测'
        speed=lambda v:f'{v:.3f}×' if v is not None else '无有效比值'
        lines.append('| '+scene['scene']+' | '+fmt(times[0])+' | '+fmt(times[1])+' | '+speed(r['graph'] if r else None)+' | '+fmt(times[2])+' | '+speed(r['all'] if r else None)+' | '+speed(r['other_components'] if r else None)+' |')
    lines += ['','求解时间为 frames.csv 的 CPU 同步包络；core event 嵌套于其中，进程 wall 含加载和不对称导出成本，详见 TIMINGS.csv。',
              '装配、线性、CCD、线搜索、状态更新是原生阶段计时，退出轮装配单列；阶段覆盖差异不能当作完整 wall 分解。',
              'Graph/base 比含活动实现与冻结原版差异，本轮无活动 host 臂，不能严格隔离纯 Graph 的因果收益。',
              '材料指标及按布料/FEM/ABD 分组的轨迹差异见 RESULTS.json；没有新基线重复标定、独立接受路径 CCD 或原版真实速度，不宣称质量已认证。',
              '来源：原生 solver commit 3d8c15e；编译、实际对象、链接、输入和配置身份见 BUILD_IDENTITY.json、SEAL.json、runs/*.json。','']
    with (ROOT/REPORT/'REPORT.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
    return {'recorded':len(rows),'result':REPORT+'/RESULTS.json'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('seal','run','report'));p.add_argument('--index',type=int)
    a=p.parse_args()
    value=seal() if a.action=='seal' else run(a.index) if a.action=='run' else report()
    if a.action!='run':print(json.dumps(value))

if __name__=='__main__':main()
