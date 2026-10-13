"""Seven historical scenes, immutable finite paper-style experiment plan."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/bench'))
from config import expand,environment

SCENES={
    'Sphere':'cloth_sphere7_l','Bunny':'cloth_fixed_bunny_l','Table':'cloth_table_l',
    'Stack':'cloth_stack10_l','Hang':'cloth_hang_l','Mixed':'bunny_cloth_bunny_l',
    'Boxes':'boxes1920_cloth_l'}
ARMS=('stiff','host','graph','pure','ours')
LABELS={'stiff':'Base (frozen StiffGIPC)','host':'Base (current host)',
    'graph':'Graph only','pure':'Pure cumulative TOI','ours':'Compensated TOI + residual gate',
    'original':'Original (Graph)','strict':'Strict reference (Graph)'}
KEYFRAMES={s:[0,25,50,75,100] for s in SCENES}
KEYFRAMES['Table']=sorted(set(KEYFRAMES['Table']+[28]))
KEYFRAMES['Bunny']=sorted(set(KEYFRAMES['Bunny']+[46]))
CHECKPOINTS={'Table':28,'Bunny':46}


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)


def config(scene,arm,steps=100,matched=False):
    source=read(ROOT/'Assets/benchmark_scenes'/f'{SCENES[scene]}.json')
    # Preserve the historical physical dt; standardize solver stopping explicitly.
    cfg=dict(scene=SCENES[scene],steps=steps,dt=source['effective_scalar_fields']['dt'],
        timeout_seconds=600,ipc_newton_tol=.01,pcg_rho_tol=1e-4,ipc_min_updates=6,
        trace_velocity=arm!='stiff',preset='base')
    if arm=='graph':cfg['preset']='graph'
    if arm in ('pure','ours') or matched:
        cfg.update(preset='combined',fem_direct_pullback=True,mas_pivot_only_sync=True,discrete_bvh_refit=True)
    if arm in ('pure','ours'):cfg['ipc_cumulative_tol']=.001
    if arm=='ours':cfg.update(ipc_termination='compensated',ipc_residual_rel_tol=.03)
    if arm=='strict':cfg.update(ipc_termination='movement_only',pcg_rho_tol=1e-6)
    return expand(cfg)


def task(scene,arm,repeat,steps=100,stage='performance',matched=False,**extra):
    return dict(name=f'{stage}_{scene.lower()}_{arm}_r{repeat}',scene_key=scene,arm=arm,repeat=repeat,
        binary='base' if arm=='stiff' else 'active',config=config(scene,arm,steps,matched),
        extra_environment={'GIPC_IPC_PCG_FORCING':'off'},**extra)


def plan():
    full=[]
    for repeat in (1,2,3):
        order=ARMS[repeat-1:]+ARMS[:repeat-1]
        scenes=list(SCENES) if repeat%2 else list(reversed(SCENES))
        for scene in scenes:
            full.extend(task(scene,arm,repeat) for arm in order)
    seed=[task(s,'original',1,CHECKPOINTS[s],stage='seed',matched=True,
        checkpoint_save_frame=CHECKPOINTS[s]) for s in CHECKPOINTS]
    matched=[]
    for s,frame in CHECKPOINTS.items():
        for repeat in (1,2,3):
            arms=('original','pure','ours','strict')
            arms=arms[repeat-1:]+arms[:repeat-1]
            for arm in arms:
                matched.append(task(s,arm,repeat,1,stage='matched',matched=True,
                    target_frame=frame,checkpoint_from=f'seed_{s.lower()}_original_r1/checkpoint'))
    return dict(schema='paper_suite.v1',scenes=SCENES,frames=100,repeats=3,arms=list(ARMS),
        labels=LABELS,keyframes=KEYFRAMES,checkpoints=CHECKPOINTS,
        stages={'smoke':[task(s,a,1,2,'smoke') for s in SCENES for a in ('stiff','ours')],
            'performance':full,'seed':seed,'matched':matched},
        definitions={
            'Hess':'sum of reported phase_ms.assembly (not just Hessian kernels)',
            'LSolver':'sum of phase_ms.pcg; includes solver preparation and execution',
            'CCD':'sum of phase_ms.ccd', 'LineS':'sum of phase_ms.line_search',
            'Misc':'TimeTot-Hess-LSolver-CCD-LineS; includes state updates, terminal assembly and other overhead',
            'TimeTot':'sum of synchronized host elapsed solver_ms, excluding initialization and trace export',
            'P_bar':'total PCG iterations / number of recorded linear solves, within each run',
            'N_bar':'recorded linear solves / 100 physical frames, within each run',
            'P_tot':'total PCG iterations over the 100 frames',
            'S':'median of three paired Base/Ours TimeTot ratios',
            'aggregation':'median of three run-level columns; raw observations retained. Independently taken medians need not sum exactly.',
            'graph_ablation':'same current binary, host vs graph; only execution changes',
            'pure':'existing legacy cumulative rule, beta threshold .001, original movement test retained',
            'ours':'existing compensated budget AND residual gate (.03), beta threshold .001; no PCG forcing',
            'matched':'one physical step from one byte-verified same-binary seed, all rows have identical combined Graph execution flags',
            'strict':'movement-only outer rule and PCG rho=1e-6, same movement threshold .01; numerical reference, not an exact solution',
            'errors':'worst candidate/reference position RMS and maximum over all 3x3 pairings, metres converted to mm',
            'reference_precision':'full trials retain MAS legacy, Float FP64, PCG rho=1e-4',
            'dt':'historical scene dt (.04 Hang, .02 Stack, .01 others), shared by every method'},
        finite_policy={'maximum_run_seconds':600,'retry':False,'failed_scene_skips_remaining_runs_in_stage':True,
            'full_performance_runs':105,'full_performance_frames':10500,'automatic_remote_execution':False},
        physical_quality_certified=False,performance_certified=False)


def env_for(t,out,exe_hash,checkpoint_root=None):
    env=environment(t['config'],out)
    env.update(t['extra_environment'],GIPC_BINARY_ID=exe_hash)
    if t['binary']=='base':
        # The frozen base recognizes only its original frontend controls.
        supported={'GIPC_SCENE','GIPC_STEPS','GIPC_DT','GIPC_NEWTON_TOL','GIPC_PCG_TOL',
            'GIPC_TRACE_DIR','GIPC_TRACE_STRIDE','GIPC_TRACE_PHYSICS','GIPC_DEFER_STATS','GIPC_DUMP_STATE','GIPC_OUTPUT_PATH'}
        env={k:v for k,v in env.items() if not k.startswith('GIPC_') or k in supported}
    if 'checkpoint_save_frame' in t:
        env.update(GIPC_CHECKPOINT_FRAME=str(t['checkpoint_save_frame']),GIPC_CHECKPOINT_DIR=str(out/'checkpoint'))
    if 'checkpoint_from' in t:
        require(checkpoint_root is not None,'Checkpoint root required')
        cp=Path(checkpoint_root)/t['checkpoint_from'];meta=read(cp/'checkpoint.json')
        require(meta['binary_id']==exe_hash and meta['next_physical_frame']==t['target_frame'],'Wrong checkpoint identity/frame')
        env['GIPC_CHECKPOINT_LOAD']=str(cp)
    return env


def input_inventory():
    result=[]
    for label,case in SCENES.items():
        source=ROOT/'Assets/benchmark_scenes'/f'{case}.json';scene=read(source)
        require(scene['case_id']==case,'Scene ID mismatch')
        files={source.relative_to(ROOT/'Assets').as_posix()}
        for o in scene['objects']:
            mesh=Path(o['stiff_mesh']);p=ROOT/'Assets'/mesh
            require(sha(p)==o['stiff_mesh_sha256'],'Frozen mesh differs: '+str(mesh))
            files.add(mesh.as_posix())
            # ABD does not run FEM MAS partitioning. FEM sorting caches must be frozen.
            if o['body_type']=='FEM':
                files.update(['sorted_mesh/'+mesh.stem+'_sorted.16'+mesh.suffix,
                    'sorted_mesh/'+mesh.stem+'_sorted.16.part'])
        records=[]
        for name in sorted(files):
            a=ROOT/'Assets'/name;b=ROOT/'baseline/Assets'/name
            require(a.is_file() and b.is_file(),'Missing common input/cache: '+name)
            require(sha(a)==sha(b),'Base/active input differs: '+name)
            records.append(dict(path=name,bytes=a.stat().st_size,sha256=sha(a)))
        result.append(dict(label=label,scene=case,files=records,dt=scene['effective_scalar_fields']['dt']))
    return result
