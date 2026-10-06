"""Sixteen explicit CPU-analyzed Linux runs; no build, cache generation or SSH.

seal --parent-seal OLD --output NEW [--pre-cache-receipt RECEIPT]
init --seal NEW --session runs/NEW_SESSION
run-next --seal NEW --session runs/NEW_SESSION [--gpu 0]
Only four-scene derived sorting caches may supplement the old runtime inputs.
"""
from __future__ import annotations
import argparse
import copy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/full_eval'), str(ROOT/'tools/bench'), str(ROOT/'tools/diagnostic')]
import analysis as metrics
import build_identity as builds
import plan as old_plan
import runner as old_runner
import linux_runner as linux
from config import read, expand, digest
from fixed_quality import execute_base
from linux_runner import child, require, sha, inventory, verify_files, write_new

CASES = {'hang_l':'cloth_hang_l', 'sphere_l':'cloth_sphere7_l',
         'hang_m':'cloth_hang_m', 'sphere_m':'cloth_sphere7_m'}
ARMS = ('original_stiff','ipc_host','ipc_graph','combined_graph')


def tasks():
    rows = []
    for index, (key, scene) in enumerate(CASES.items()):
        for arm in ARMS if index % 2 == 0 else reversed(ARMS):
            item = old_plan.task('hang', arm, 'paired', 1)
            item.update(name=f'p1_{key}_{arm}',scene_key=key,index=len(rows)+1)
            item['config']['scene'] = scene
            item['expanded_config_sha256'] = digest(expand(item['config']))
            rows.append(item)
    return rows


def plan():
    return {'schema':'resolution_eval_plan.v1','tasks':tasks(),'maximum_runs':16,
            'frames':100,'timeout_seconds':120,'from_zero':True,'automatic_retry':False,
            'one_task_per_invocation':True,'stop_scope':'Remaining arms of the failed case',
            'quality_certified':False,'performance_certified':False,
            'scope':'One sample per case/arm; observed ratios and material/state differences, no statistical certification.'}


def asset_proof(root, previous, current):
    """Validate the exact additive exception, without excusing an old mutation."""
    old = {r['path']:r for r in previous}; now = {r['path']:r for r in current}
    require(len(old)==len(previous) and len(now)==len(current), 'Duplicate source identity')
    require(all(now.get(name)==row for name,row in old.items()), 'Old source/input missing or changed')
    active = {p[len('Assets/'):]:r for p,r in now.items() if p.startswith('Assets/')}
    base = {p[len('baseline/Assets/'):]:r for p,r in now.items() if p.startswith('baseline/Assets/')}
    require(active.keys() <= base.keys(), 'Active-only asset')
    require(all((r['bytes'],r['sha256'])==(base[p]['bytes'],base[p]['sha256']) for p,r in active.items()), 'Common assets differ')
    require(base.keys()-active.keys() <= builds.UNUSED_BASE_CACHES, 'Unknown baseline-only asset')
    allowed = set(); references = []
    for case in CASES.values():
        relative = f'benchmark_scenes/{case}.json'
        require(relative in active, 'Missing declared scene')
        scene = read(child(root,'Assets/'+relative))
        require(scene.get('case_id')==case and isinstance(scene.get('objects'),list) and scene['objects'], 'Scene identity/objects differ')
        meshes = []
        for obj in scene['objects']:
            name = obj['stiff_mesh']; mesh = child(root/'Assets',name)
            require(name in active and name not in builds.UNUSED_BASE_CACHES
                    and active[name]['sha256']==obj['stiff_mesh_sha256'], 'Referenced mesh identity differs')
            dimension,body_type=obj['dimension'],obj['body_type']
            require(dimension in (2,3) and body_type in ('FEM','ABD'),'Unsupported scene geometry identity')
            # This plan fixes native preconditionerType=1 (MAS) in all four arms.
            # SimpleSceneImporter loads 3D ABD tetrahedra directly, without METIS.
            needs_cache=dimension==2 or (dimension==3 and body_type=='FEM')
            caches = [f'sorted_mesh/{mesh.stem}_sorted.16{mesh.suffix}',f'sorted_mesh/{mesh.stem}_sorted.16.part'] if needs_cache else []
            require(not (set(caches)&builds.UNUSED_BASE_CACHES), 'Selected scene references an excluded baseline cache')
            require(all(p in active and p in base for p in caches), 'Both asset trees need every selected derived cache')
            allowed.update(prefix+p for prefix in ('Assets/','baseline/Assets/') for p in caches)
            meshes.append({'mesh':name,'mesh_sha256':active[name]['sha256'],'dimension':dimension,
                           'body_type':body_type,'preconditioner_type':1,'cache_required':needs_cache,
                           'caches':[{k:active[p][k] for k in ('path','bytes','sha256')} for p in caches]})
        references.append({'scene':case,'scene_sha256':active[relative]['sha256'],'meshes':meshes})
    additions = now.keys()-old.keys()
    require(additions <= allowed, 'Unexpected new source/input outside selected derived caches')
    return {'passed':True,'added_files':[now[p] for p in sorted(additions)],'selected_scene_references':references,
            'common_assets':len(active),'baseline_only_unchanged':sorted(base.keys()-active.keys()),
            'scope':'Old bytes unchanged. Only selected derived .16 runtime caches added; executables are reused, not rebuilt.'}


def parent_inputs(root, parent_path, expected_sha=None):
    if expected_sha is not None: require(sha(parent_path)==expected_sha,'Parent seal changed')
    parent = read(parent_path)
    require(parent['schema']=='full_eval_seal.v1' and digest(parent['plan'])==parent['plan_sha256'], 'Invalid parent seal')
    require(old_runner.controller_files(root)==parent['controller_files']
            and digest(parent['controller_files'])==parent['controller_sha256'], 'Old controller identity changed')
    verify_files(root,parent['controller_files'])
    active_path = child(root,parent['active_manifest_path'])
    require(sha(active_path)==parent['active_manifest_sha256']
            and read(active_path)==parent['active_manifest'],'Original active manifest changed')
    attachment = parent['build_identity_attachment']; path = child(root,attachment['path'])
    require(sha(path)==attachment['sha256'],'Actual build attachment changed')
    packet = read(path)
    require(packet['schema']=='full_eval_linux_build_identity.v1' and packet['passed'] is True
            and packet['sources_unchanged'] is True and packet['temporal_freshness']['passed'] is True
            and packet['assets_equivalence']['passed'] is True, 'Original build proof not passing')
    # This verifier checks every original source/object/link/compiler record. It
    # deliberately does not compare the current source set: the sole exception
    # to that old set is checked immediately below, not silently ignored.
    builds._verify_identity_files(root,packet)
    old_runner.match_build_programs(packet,parent['active_manifest'],parent['base_manifest'])
    current = builds.source_inventory(root)
    proof = asset_proof(root,packet['sources'],current)
    active_current = linux.source_inventory(root)
    old_active = {r['path']:r for r in parent['active_manifest']['sources']}
    new_active = {r['path']:r for r in active_current}
    added_active = {r['path'] for r in proof['added_files'] if r['path'].startswith('Assets/')}
    require(all(new_active.get(p)==r for p,r in old_active.items())
            and new_active.keys()-old_active.keys()==added_active, 'Active source/tool set differs beyond new caches')
    manifests = {}
    for kind in ('active','base'):
        original = parent[kind+'_manifest']
        require(digest(original['sources'])==original['source_digest'],'Original source digest differs')
        verify_files(root,original['sources']); verify_files(root,original['build_evidence'])
        require(sha(child(root,original['exe_path']))==original['exe_sha256'],'Original executable changed')
        extras = [r for r in proof['added_files'] if r['path'].startswith('Assets/' if kind=='active' else 'baseline/Assets/')]
        manifest = copy.deepcopy(original)
        manifest['sources'] = sorted(original['sources']+extras,key=lambda r:r['path'])
        manifest['source_digest'] = digest(manifest['sources'])
        manifest['runtime_input_supplement'] = {'original_source_digest':original['source_digest'],
            'added_files':extras,'recompiled':False,'scope':'Same executable/object/link identity with explicitly sealed additional runtime sorting caches.'}
        manifests[kind+'_manifest'] = manifest
    return manifests, proof


def tools_identity(root):
    # Offline report formatting is not part of the executable controller seal.
    # Its separate source identity belongs in the generated offline report.
    return inventory(root,[child(root,'tools/resolution_eval/'+name) for name in ('run.py','run_test.py')])


def seal(root,parent_name,output,receipt=None,cache_receipt=None):
    path = child(root,parent_name); manifests,proof = parent_inputs(root,path)
    value = {'schema':'resolution_eval_seal.v1','parent_seal_path':parent_name,'parent_seal_sha256':sha(path),
             'plan':plan(),'plan_sha256':digest(plan()),'runtime_input_proof':proof,
             'tools':tools_identity(root),**manifests}
    if receipt: value['pre_cache_receipt'] = {'path':receipt,'sha256':sha(child(root,receipt))}
    if cache_receipt: value['cache_preparation_receipt'] = {'path':cache_receipt,'sha256':sha(child(root,cache_receipt))}
    write_new(child(root,output),value)
    return value


def verify_seal(root,path):
    value = read(path)
    require(value['schema']=='resolution_eval_seal.v1' and value['plan']==plan()
            and value['plan_sha256']==digest(plan()) and value['tools']==tools_identity(root),'Resolution plan/tools changed')
    manifests,proof = parent_inputs(root,child(root,value['parent_seal_path']),value['parent_seal_sha256'])
    require(proof==value['runtime_input_proof'] and all(value[k]==v for k,v in manifests.items()),'Supplemented runtime inputs changed')
    for key in ('pre_cache_receipt','cache_preparation_receipt'):
        if key in value:
            receipt=value[key];require(sha(child(root,receipt['path']))==receipt['sha256'],'Cache preparation/verification receipt changed')
    return value


def initialize(root,seal_name,session_name):
    value=verify_seal(root,child(root,seal_name));session=old_runner.session_path(root,session_name)
    require(not session.exists(),'New resolution session required; never overwrite')
    session.mkdir(parents=True)
    write_new(session/'session.json',{'seal_path':seal_name,'seal_sha256':sha(child(root,seal_name)),'plan_sha256':digest(plan())})
    write_new(session/'plan.json',value['plan'])
    return {'status':'initialized','maximum_runs':16}


def entries(root,session):
    result=[];previous=None
    for index,path in enumerate(sorted(child(session,'ledger').glob('*.json')),1):
        require(index<=16 and path.name==f'{index:04d}.json' and not path.is_symlink(),'Ledger order/link differs')
        row=read(path);require(row['task']==tasks()[index-1] and row['predecessor_sha256']==previous,'Task/predecessor differs')
        for key in ('analysis','summary'):
            require(sha(child(session,row[key+'_path']))==row[key+'_sha256'],key+' artifact changed')
        require(read(child(session,row['analysis_path']))==row['analysis'],'Embedded analysis differs')
        if row['run_created']:
            out=child(session,row['task']['name'])
            require(sha(out/'evidence.json')==row['evidence_sha256'] and read(out/'result.json')==row['result'],'Run metadata changed')
            old_runner.verify_retained_evidence(root,out,row['evidence_sha256'])
        for pair in row['analysis'].get('paired_comparisons',[]):
            require(sha(child(session,pair['pair_cache_path']))==pair['pair_cache_sha256'],'Pair comparison cache changed')
        result.append(row);previous=sha(path)
    return result


def summarize(rows):
    report={'schema':'resolution_eval_summary.v1','recorded_tasks':len(rows),'planned_tasks':16,
            'quality_certified':False,'performance_certified':False,'statistical_certification':False,'cases':{},
            'scope':'One sample per arm. Single-baseline signed material differences are observations, not frozen repeat ranges or equivalence.'}
    for case in CASES:
        good={r['task']['arm']:r['analysis'] for r in rows if r['task']['scene_key']==case and r['status']=='completed'}
        result={'completed_arms':list(good),'ratios':{},'material_vs_single_original':{},
                'paired_states':[p for v in good.values() for p in v.get('paired_comparisons',[])]}
        for left,right in (('original_stiff','combined_graph'),('original_stiff','ipc_host'),('ipc_host','ipc_graph'),('ipc_graph','combined_graph')):
            if left in good and right in good:
                a,b=good[left],good[right]
                same=metrics.compare_inputs(a['input_identity'],b['input_identity'])['passed'] and a['gpu_uuid']==b['gpu_uuid']
                result['ratios'][left+'/'+right]={'comparable':same,'solver':a['timing']['solver_seconds']/b['timing']['solver_seconds'] if same else None,
                    'process_wall':a['timing']['process_wall_seconds']/b['timing']['process_wall_seconds'] if same else None,'samples':1}
        if 'original_stiff' in good:
            baseline=good['original_stiff']['material']
            bound={'bounds':{k:{'applicable':baseline[k] is not None,'min':baseline[k],'max':baseline[k],'worse_direction':d}
                             for k,d in metrics.MATERIAL_DIRECTIONS.items()},'scope':'Single original-Stiff observation; not calibration'}
            bound['sha256']=digest(bound)
            result['material_vs_single_original']={arm:metrics.compare_material(r['material'],bound) for arm,r in good.items() if arm!='original_stiff'}
        report['cases'][case]=result
    return report


def stiff_host_comparison(session,task,row):
    """Add the one edge absent from the unchanged full-evaluation analyzer."""
    left,right='original_stiff','ipc_host'
    if task['arm'] not in (left,right):return
    other_arm=right if task['arm']==left else left
    other_name=f'p1_{task["scene_key"]}_{other_arm}'
    prior_path=child(session,f'analysis/{other_name}.json')
    if not prior_path.is_file():return
    prior=read(prior_path)
    if not prior.get('hard_checks_passed'):return
    inputs=metrics.compare_inputs(row['input_identity'],prior['input_identity'])
    require(inputs['passed'],'Original/host physical inputs differ')
    names={task['arm']:task['name'],other_arm:other_name}
    summaries={task['arm']:row,other_arm:prior}
    folders={arm:child(session,names[arm]) for arm in (left,right)}
    sources={side:{'run':folders[arm].name,'evidence_sha256':sha(folders[arm]/'evidence.json')}
             for side,arm in (('left',left),('right',right))}
    path=child(session,f'pair_analysis/p1_{task["scene_key"]}_{left}__{right}.json')
    record={'left':left,'right':right,'pair':1,'scene_key':task['scene_key'],'initial_inputs':inputs,
            'input_evidence':sources,'schema':'full_eval_pair_states.v1'}
    complete=True
    for arm,folder in folders.items():
        kinds=['state']+(['velocity'] if summaries[arm].get('actual_velocity',{}).get('available') else [])
        complete &= all((folder/'trace'/f'{kind}_{frame:04d}.bin').is_file() for kind in kinds for frame in range(101))
    if complete:
        for folder in folders.values():
            raw=[r for r in read(folder/'evidence.json')['files'] if r['path'].startswith(('trace/state_','trace/velocity_'))
                 or r['path'] in ('trace/topology.bin','trace/metadata.json')]
            verify_files(folder,raw)
        record['state_difference']=metrics.state_comparison(folders[left],folders[right])
    else:record['state_difference']={'available':False,'reason':'Complete paired raw exports unavailable; no state/velocity difference reconstructed.'}
    write_new(path,record)
    row.setdefault('paired_comparisons',[]).append(dict(record,pair_cache_path=path.relative_to(session).as_posix(),pair_cache_sha256=sha(path)))


def advance(root,seal_name,session_name,gpu=0):
    path=child(root,seal_name);value=verify_seal(root,path);session=old_runner.session_path(root,session_name)
    require(read(child(session,'session.json'))=={'seal_path':seal_name,'seal_sha256':sha(path),'plan_sha256':digest(plan())}
            and read(child(session,'plan.json'))==plan(),'Session identity changed')
    with linux.gpu_lock(root):
        old=entries(root,session)
        require(not any(r.get('analysis_exception') for r in old),'Prior analysis/controller exception; no retry')
        if len(old)==16:return {'status':'finished','gpu_launched':False}
        task=tasks()[len(old)];index=task['index'];out=child(session,task['name'])
        require(not out.exists(),'Unledgered run exists; do not retry or overwrite')
        row={'task':task,'index':index,'predecessor_sha256':sha(session/'ledger'/f'{index-1:04d}.json') if old else None,
             'run_created':False,'gpu_launched':False,'analysis_exception':False}
        if any(r['task']['scene_key']==task['scene_key'] and r['status']=='hard_failed' for r in old):
            result={'status':'skipped','reason':'This case had a hard failure; remaining arms are not launched.'}
            check={'name':task['name'],'hard_checks_passed':False,'quality_status':'not_run','failures':[result['reason']]};row['status']='skipped'
        else:
            try:
                manifest=value[task['binary']+'_manifest']
                result=(execute_base if task['binary']=='base' else linux.execute)(root,session,task,manifest,gpu)
                row['run_created']=True;row['gpu_launched']=(out/'process.json').is_file()
                row['evidence_sha256']=sha(out/'evidence.json')
                check=metrics.analyze_run(session,task,value)
                require(type(check.get('hard_checks_passed')) is bool,'Missing boolean CPU hard gate')
                if check['hard_checks_passed']:
                    try:stiff_host_comparison(session,task,check)
                    except (ValueError,KeyError,FileNotFoundError) as exc:
                        check=dict(check,hard_checks_passed=False,failures=list(check.get('failures',[]))+[str(exc)])
                if result['status']=='completed' and (not metrics.finite(result.get('wall_seconds')) or not 0<result['wall_seconds']<=120):
                    check=dict(check,hard_checks_passed=False,failures=list(check.get('failures',[]))+['Absolute 120-second deadline exceeded.'])
                row['status']='completed' if result['status']=='completed' and check['hard_checks_passed'] else 'hard_failed'
            except Exception as exc:
                row.update(status='hard_failed',analysis_exception=True)
                result=read(out/'result.json') if (out/'result.json').is_file() else {'status':'controller_failed'}
                check={'name':task['name'],'hard_checks_passed':False,'quality_status':'analysis_failed','failures':[type(exc).__name__+': '+str(exc)]}
                row['run_created']=(out/'result.json').is_file() and (out/'evidence.json').is_file()
                if row['run_created']:row['evidence_sha256']=sha(out/'evidence.json')
        row.update(result=result,analysis=check,analysis_path=f'analysis/{task["name"]}.json',summary_path=f'summary/{index:04d}.json')
        write_new(child(session,row['analysis_path']),check);row['analysis_sha256']=sha(child(session,row['analysis_path']))
        try: summary=summarize(old+[row])
        except Exception as exc:
            row['analysis_exception']=True;summary={'error':type(exc).__name__+': '+str(exc),'quality_certified':False,'performance_certified':False}
        write_new(child(session,row['summary_path']),summary);row['summary_sha256']=sha(child(session,row['summary_path']))
        write_new(child(session,f'ledger/{index:04d}.json'),row)
        return {k:row[k] for k in ('status','index','gpu_launched','analysis_exception')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('seal');p.add_argument('--parent-seal',required=True);p.add_argument('--output',required=True);p.add_argument('--pre-cache-receipt');p.add_argument('--cache-preparation-receipt')
    for name in ('init','run-next'):
        p=subs.add_parser(name);p.add_argument('--seal',required=True);p.add_argument('--session',required=True)
        if name=='run-next':p.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='seal':seal(root,args.parent_seal,args.output,args.pre_cache_receipt,args.cache_preparation_receipt);result={'status':'sealed'}
    elif args.command=='init':result=initialize(root,args.seal,args.session)
    else:result=advance(root,args.seal,args.session,args.gpu)
    import json
    print(json.dumps(result));return 2 if result.get('analysis_exception') else 0


if __name__=='__main__':raise SystemExit(main())
