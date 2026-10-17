"""Account measured nested windows and verify why old/new iteration counts differ."""
import json
from pathlib import Path
from m3_numeric import ROOT,need,rel
import numpy as np
from cpu_fixed_reference import load_snapshot
from local_identity import record,verify
from linux_runner import write_new
from analyze_native_balanced import csv_new

RAW=ROOT/'runs/innovation_p5_native_balanced_20261016'
REPORT=ROOT/'reports/innovation_p5_native_balanced_20261016'
OLD=ROOT/'runs/innovation_p5_native_history_20261016'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    destination=REPORT/'COST_AND_CROSS_BATCH_AUDIT.json'
    need(not destination.exists(),'Do not overwrite cost audit')
    pairs=[];quality=[];changes=[];setup=[];inputs=[record(RAW/'analysis/summary.json')]
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        path=RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json';study=read(path)
        source_path=RAW/label/'fixed'/f'f{frame}_n1_native_balanced_study.json';source=read(source_path)
        inputs += [record(path),record(source_path)]
        for pair in (1,2,3):
            base=next(r for r in study['runs'] if r['pair']==pair and not r['requested_balanced'])
            aug=next(r for r in study['runs'] if r['pair']==pair and r['requested_balanced'])
            delta=aug['solve_host_ms']-base['solve_host_ms']
            qdelta=aug['result']['quality_host_ms']-base['result']['quality_host_ms']
            pairs.append(dict(scene=scene,pair=pair,solve_delta_ms=delta,
                measured_quality_delta_ms=qdelta,remainder_delta_ms=delta-qdelta,
                subtraction_is_accounting_not_production_timing=True))
        for row in study['runs']:
            result=row['result'];checked=[p for p in result['trace'] if p['quality_checked']]
            need(checked and checked[-1]['iteration']==result['iterations'],'Missing final quality')
            quality.append(dict(scene=scene,pair=row['pair'],balanced=row['used_balanced'],
                iterations=result['iterations'],first_quality_iteration=checked[0]['iteration'],
                quality_checks=result['quality_checks'],first_quality_l2=checked[0]['true_relative_l2'],
                first_quality_weighted=checked[0]['common_native_MAS_relative'],
                quality_ms=result['quality_host_ms']))
        setup.append(dict(scene=scene,source_build_ms=source['source_build_host_ms'],
            source_control_ms=source['runs'][0]['solve_host_ms'],source_capture_ms=source['runs'][1]['solve_host_ms'],
            source_export_ms=source['runs'][1]['export_and_residual_host_ms'],
            target_coarse_ms=study['target_coarse_host_ms'],
            coarse_invocations=1,cold_warm_separation_measured=False,
            diagnostic_target_also_allocates_unused_history_bank=True))
        for direction in (1,2):
            previous=OLD/label/'fixed'/f'f{frame}_n{direction}'
            current=RAW/label/'fixed'/f'f{frame}_n{direction}'
            old_A,old_b,old_active,*_=load_snapshot(previous)
            new_A,new_b,new_active,*_=load_snapshot(current)
            inputs += [record(str(previous)+'_meta.json'),record(str(current)+'_meta.json')]
            row=dict(scene=scene,direction=direction,same_dimensions=old_A.shape==new_A.shape,
                     old_nnz=old_A.nnz,new_nnz=new_A.nnz,
                     same_active_coordinates=np.array_equal(old_active,new_active))
            if row['same_dimensions']:
                difference=new_A-old_A
                row.update(A_relative_frobenius=float(np.linalg.norm(difference.data)/np.linalg.norm(old_A.data)),
                           b_relative_l2=rel(new_b,old_b),identical_A=difference.nnz==0,
                           identical_b=np.array_equal(new_b,old_b))
            changes.append(row)
    for name,rows in (('paired_cost_accounting',pairs),('quality_schedule',quality),('cross_batch_systems',changes),('setup_costs',setup)):
        csv_new(REPORT/'tables'/(name+'.csv'),rows)
    verify(inputs)
    means=[dict(scene=scene,**{key:float(np.mean([r[key] for r in pairs if r['scene']==scene]))
        for key in ('solve_delta_ms','measured_quality_delta_ms','remainder_delta_ms')}) for scene in ('Bunny','Mixed')]
    write_new(destination,dict(passed=True,source=record(Path(__file__)),inputs=inputs,
        pairs=pairs,additive_pair_means=means,quality_schedule=quality,cross_batch_systems=changes,setup=setup,
        causal_isolation_of_projection_cost=False,cold_warm_coarse_measured=False,performance_certified=False))
    print(json.dumps(dict(passed=True,additive_pair_means=means,quality_schedule=quality,cross_batch_systems=changes)),flush=True)

if __name__=='__main__':main()
