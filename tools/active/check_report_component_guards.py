"""CPU-only binding of local component audits before the long diagnostic matrix."""
import json
import numpy as np
from config import ROOT, read, sha
from ipc_benchmark import write
from report_components import TAG
from verify_systems import accurate_reference, matrix_from_snapshot


def main():
    folder=ROOT/'reports/active'
    smoke=read(folder/f'{TAG}_smoke_batch.json')
    guards=read(folder/f'{TAG}_guards_batch.json')
    assert len(smoke['runs'])==12 and len(guards['runs']) in (5,6)
    assert all(not row['hard_failures'] for row in smoke['runs'] if row['scene_key']!='mixed')
    assert all(not row['hard_failures'] or row['result']['status']=='memory_budget' for row in guards['runs'])
    dot=[];trees=[]
    for row in guards['runs']:
        run=ROOT/'runs/active'/row['name']
        completed=row['result']['status']=='completed'
        if completed:assert read(run/'component_execution_checks.json')['passed']
        else:
            assert row['result']['status']=='memory_budget' and row['config']['discrete_bvh_validate']
            requested=read(run/'requested.json')['expanded_config']
            resolved=read(run/'resolved_config.json')['report_components']
            assert requested['discrete_bvh_refit'] and resolved['discrete_bvh_refit'] and resolved['discrete_bvh_validate']
            # Validate only completed exported prefix states. This does not
            # reclassify the requested 59-frame resource failure as completed.
            count=row['result']['recorded_frames'];assert count>0
            expected=np.fromfile(run/'trace/state_0000.bin',dtype='<f8').shape
            for i in range(count+1):
                values=np.fromfile(run/f'trace/state_{i:04d}.bin',dtype='<f8')
                assert values.shape==expected and np.isfinite(values).all()
        if row['config'].get('fixed_mas_dot_study'):
            files=list((run/'fixed').glob('*_mas_dot_study.json'))
            assert len(files)==1
            study=read(files[0])
            assert study['passed']
            for key in ('all_work_buffers_restored_bitwise','rhs_unchanged_bitwise',
                        'system_and_MAS_scratch_unchanged','production_graph_unchanged'):
                assert study[key]
            assert len(study['cases'])==3 and all(len(c['runs'])==3 for c in study['cases'])
            assert all(r['passed'] and r['z_bitwise_equal'] and r['rho_absolute_error']<=c['rho_roundoff_bound']
                       for c in study['cases'] for r in c['runs'])
            assert len(study['full_fused_PCG_zero_rhs'])==2
            assert all(r['passed'] and r['iterations']==0 and r['x_z_r_finite_zero'] for r in study['full_fused_PCG_zero_rhs'])
            prefix=run/'fixed/f2_n1'
            meta,matrix,rhs=matrix_from_snapshot(prefix)
            _,reference=accurate_reference(matrix,rhs)
            assert reference['passed']
            dot.append({'run':row['name'],'passed':True,'case_count':3,'graph_and_host_comparisons':9,
                'full_fused_zero_rhs_pcg_modes':2,'cpu_reference':reference,'study_sha256':sha(files[0])})
        else:
            frames=read(run/'output/stats.json')['frames']
            counts={kind:{key:sum(f['discrete_bvh'][kind][key] for f in frames) for key in
                    ('validation_calls','validation_passed','validation_failed','diagnostic_refits',
                     'diagnostic_pairs_compared','diagnostic_overflow_retries','production_refits','swept_rebuilds')}
                    for kind in ('face','edge')}
            assert all(c['validation_calls']>0 and c['validation_passed']==c['validation_calls']
                       and c['validation_failed']==0 and c['diagnostic_refits']>0 for c in counts.values())
            trees.append({'run':row['name'],'steps':row['config']['steps'],
                'completed_prefix_frames':row['result']['recorded_frames'],'run_status':row['result']['status'],
                'same_state_checks_passed_on_recorded_prefix':True,'full_requested_window_passed':completed,'counts':counts})
    assert len(dot)==2 and len(trees) in (3,4)
    contact=next(t for t in trees if t['steps']==59)
    assert sum(c['diagnostic_pairs_compared'] for c in contact['counts'].values())>0, 'Nonempty contact validation not covered'
    assert sum(c['swept_rebuilds'] for t in trees for c in t['counts'].values())>0, 'Swept-to-discrete invalidation not covered'
    fixtures=read(ROOT/f'runs/active/{TAG}_fixtures/result.json')
    assert fixtures['passed'] and len(fixtures['checks'])==8
    identity=sha(ROOT/'builds/active/Release/gipc.exe')
    assert identity==fixtures['exe_sha256']
    assert all(read(ROOT/'runs/active'/row['name']/'requested.json')['exe_sha256']==identity
               for row in smoke['runs']+guards['runs'])
    report={'passed':True,'scope':'Cloth component guards only; mixed resource failures do not pass',
            'smoke_runs':12,'cloth_smoke_completed':10,'local_guard_runs':len(guards['runs']),
            'completed_guard_runs':sum(r['result']['status']=='completed' for r in guards['runs']),
            'full_guard_windows_completed':all(r['result']['status']=='completed' for r in guards['runs']),
            'existing_gpu_fixtures_passed':8,
            'exe_sha256':identity,'MAS_dot_audits':dot,'BVH_same_state_audits':trees,
            'limitations':['Default PCG rho is not a true residual target.',
                'Frozen collect equality does not imply repeated full legacy atomic MAS equality.',
                'Guard refits and their tree age are diagnostic; production hit rate is measured separately.',
                'No explicit topology content edit or production Graph capacity growth fixture this round.',
                'Mixed compatibility is unavailable when the existing memory reserve stops its smoke; no retry.'],
            'quality_certified':False,'performance_certified':False,'default_promoted':False}
    write(folder/f'{TAG}_guard_checks.json',report)
    print(json.dumps({'passed':True,'dot_systems':len(dot),'bvh_runs':len(trees),
        'nonempty_pairs_compared':sum(c['diagnostic_pairs_compared'] for c in contact['counts'].values()),
        'cpu_reference_max_residual':max(d['cpu_reference']['true_relative_residual'] for d in dot)}))


if __name__=='__main__':main()
