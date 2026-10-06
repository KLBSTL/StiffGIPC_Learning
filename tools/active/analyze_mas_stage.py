"""Independent verification of the one-system MAS stage attribution."""
import json
from pathlib import Path
import numpy as np
from config import ROOT, read, sha, matches_requested
from ipc_benchmark import write
from verify_systems import matrix_from_snapshot, accurate_reference

TAG='ipc_mas_stage_20261005'


def fnv(data):
    value=14695981039346656037
    for byte in data:
        value=((value ^ byte)*1099511628211) & ((1<<64)-1)
    return value


def main():
    folder=ROOT/'reports/active'
    protocol=read(folder/f'{TAG}_protocol.json')
    batch=read(folder/f'{TAG}_batch.json')
    plan=read(ROOT/f'configs/active/{TAG}.json')
    assert sha(ROOT/f'configs/active/{TAG}.json')==protocol['plan_sha256']==batch['plan_sha256']
    assert sha(folder/'ipc_revision_20261005_quality_protocol.json')==protocol['old_quality_protocol_sha256']
    checks=[]
    for spec in plan['runs']:
        run=ROOT/'runs/active'/spec['name'];req=read(run/'requested.json');out=read(run/'result.json')
        resolved=read(run/'resolved_config.json');c=req['expanded_config']
        assert matches_requested(c,spec['config'])
        assert resolved['fixed_mas_stage_study']==c['fixed_mas_stage_study']
        assert resolved['ipc_stopping']['termination']=='legacy' and resolved['pcg_rho_tol']==1e-4
        assert out['status']=='completed' and out['finite'] and out['recorded_frames']==2
        assert sha(Path(req['command'][0]))==req['exe_sha256']
        stats=read(run/'output/stats.json')['frames']
        pcg=[n['pcg'] for f in stats for n in f['newton'] if 'pcg' in n]
        assert not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg)
        if not c['fixed_mas_stage_study']:
            assert not list((run/'fixed').glob('*_study.json'))
        checks.append({'run':run.name,'passed':True,'exe_sha256':req['exe_sha256'],
                       'production_pcg':[p['iterations'] for p in pcg],
                       'fixed_mas_stage_study':c['fixed_mas_stage_study']})
    run=ROOT/'runs/active'/plan['runs'][0]['name'];prefix=run/'fixed/f2_n1'
    study=read(Path(str(prefix)+'_study.json'));probe=study['mas_stage_probe']
    assert study['stage_only'] and not study['runs']
    restoration={k:study[k] for k in ['system_unchanged','full_snapshot_unchanged_including_scratch',
        'primary_restored_bitwise','z_restored_bitwise','graph_signature_restored']}
    assert all(restoration.values()) and probe['input_r_unchanged'] and probe['private_buffer_addresses_unchanged']
    assert probe['stage_replays']==16 and probe['warmups_per_stage']==1 and probe['observations_per_stage']==3
    locals=[m for m in study['system']['local_preconditioners'] if m['kind']=='MAS_full_owned_buffers']
    assert len(locals)==1
    local=locals[0];nodes,mapped,levels,_,clusters,*_=local['dimensions']
    input_path=Path(probe['input_r_file']);input_r=np.fromfile(input_path,dtype='<f8')
    rhs=np.fromfile(Path(str(prefix)+'_rhs.bin'),dtype='<f8')
    # Local offsets are 3-vector block offsets, not scalar DOF offsets.
    offset=local['offset']*3;assert np.array_equal(input_r,rhs[offset:offset+nodes*3])
    assert fnv(input_path.read_bytes())==probe['input_r_fnv1a64']
    records=[]
    for stage in probe['stages']:
        assert len(stage['outputs'])==4 and stage['upstream_fixed']
        dtype='<f4' if stage['output_dtype']=='float32' else '<f8'
        ref=None;observations=[]
        for item in stage['outputs']:
            path=Path(item['file']);data=path.read_bytes();values=np.frombuffer(data,dtype=dtype)
            assert np.isfinite(values).all() and fnv(data)==item['fnv1a64']
            assert values.size==(clusters if dtype=='<f4' else nodes)*3
            if item['warmup']:
                ref=values.astype(np.float64);continue
            delta=values.astype(np.float64)-ref
            nz=np.flatnonzero(delta)
            item_result={'repeat':item['repeat'],'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(ref)),
                'max_abs':float(np.abs(delta).max()),'different_components':int(nz.size),
                'first_changed_components':nz[:12].tolist(),'file_sha256':sha(path)}
            if stage['stage']=='restrict':
                item_result['fine_different_components']=int(np.count_nonzero(delta[:mapped*3]))
                item_result['coarse_different_components']=int(np.count_nonzero(delta[mapped*3:]))
            observations.append(item_result)
        records.append({'stage':stage['stage'],'observations':observations,
            'varying':any(o['different_components'] for o in observations)})
    assert [r['stage'] for r in records]==['restrict','local','prolong','full_action']
    _,a,b=matrix_from_snapshot(prefix);_,cpu=accurate_reference(a,b)
    assert cpu['passed']
    manifest=read(run/'build_manifest.json')
    identity=[]
    for unit in manifest['compiler_inputs']:
        for kind in ('source','object'):
            rec=unit[kind];identity.append({'kind':kind,'path':rec['path'],
                'passed':sha(ROOT/rec['path'])==rec['sha256']})
    assert all(r['passed'] for r in identity)
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    disabled=ROOT/'runs/active'/plan['runs'][1]['name']
    prior=ROOT/'runs/active/ipc_observer_fixed_probe_20261005_f2'
    prefix_checks=[]
    for other in (disabled,prior):
        assert (run/'trace/state_0000.bin').read_bytes()==(other/'trace/state_0000.bin').read_bytes()
        frame_differences=[]
        for frame in (1,2):
            x=np.fromfile(run/f'trace/state_{frame:04d}.bin',dtype='<f8')
            y=np.fromfile(other/f'trace/state_{frame:04d}.bin',dtype='<f8')
            frame_differences.append({'frame':frame,'position_component_rms_m':float(np.sqrt(np.mean((x-y)**2))),
                                      'position_max_abs_component_m':float(np.abs(x-y).max())})
        prefix_checks.append({'other':other.name,'initial_state_identical':True,'frames':frame_differences,
            'scope':'Short-prefix observation only; not 100-frame equivalence or quality certification'})
    report={'passed':True,'scope':'Single fixed-system stage attribution; not performance/trajectory quality acceptance',
        'run_checks':checks,'restoration_checks':restoration,'operator_input_matches_rhs':True,
        'dimensions':{'nodes':nodes,'mapped_nodes':mapped,'levels':levels,'clusters':clusters},
        'stages':records,'cpu_linear_reference':cpu,'build_identity_checks':identity,
        'short_prefix_comparisons':prefix_checks,
        'old_quality_protocol_unchanged':True,'production_tolerances_changed':False,
        'limits':'Three observations do not prove a stage always deterministic. Restriction variation on f2 cannot by itself prove it causes all long-window material differences.',
        'protocol_sha256':sha(folder/f'{TAG}_protocol.json')}
    write(folder/f'{TAG}_analysis.json',report)
    print(json.dumps({'passed':True,'stages':[{'stage':r['stage'],'varying':r['varying'],
        'max_relative_l2':max(o['relative_l2'] for o in r['observations'])} for r in records],
        'cpu_true_relative_residual':cpu['true_relative_residual'],'restoration':restoration}))


if __name__=='__main__':main()
