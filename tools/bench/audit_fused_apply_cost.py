"""Cross-batch input and unchanged-stage context; not a causal timing claim."""
import json
from pathlib import Path
from m3_numeric import ROOT,FrozenSystem,need,rel
import numpy as np
from local_identity import record,verify
from linux_runner import write_new
from analyze_native_balanced import csv_new

RAW=ROOT/'runs/innovation_p5_fused_apply_20261016'
OLD=ROOT/'runs/innovation_p5_quality_screen_20261016'
REPORT=ROOT/'reports/innovation_p5_fused_apply_20261016'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
    dest=REPORT/'COST_CONTEXT_AUDIT.json';need(not dest.exists(),'Fresh context audit required')
    summary=read(RAW/'analysis/summary.json');need(summary['implementation_checks_passed'],'Numerical audit required')
    parent=read(REPORT/'PRIOR_SOURCES.json')['parent'];verify([parent])
    sealed={r['path']:r for r in read(parent['path'])['files']}
    current={r['path']:r for r in summary['inputs']};inputs=[parent];systems=[];stages=[]
    for label,scene,frame in (('bunny','Bunny',14),('mixed','Mixed',15)):
        for direction in (1,2):
            prefixes=[folder/label/'fixed'/f'f{frame}_n{direction}' for folder in (OLD,RAW)]
            for prefix,known in zip(prefixes,(sealed,current)):
                for path in prefix.parent.glob(prefix.name+'*'):
                    if not path.is_file():continue
                    item=record(path);need(known.get(item['path'])==item,'Context input differs from original seal');inputs.append(item)
            old,new=map(FrozenSystem,prefixes)
            row=dict(scene=scene,direction=direction,old_dofs=len(old.full_b),new_dofs=len(new.full_b),
                shape_equal=old.full_A.shape==new.full_A.shape,active_equal=np.array_equal(old.active,new.active))
            if row['shape_equal']:
                delta=new.full_A-old.full_A
                row.update(A_relative_frobenius=float(np.linalg.norm(delta.data)/max(np.linalg.norm(old.full_A.data),1e-300)),
                           b_relative_l2=rel(new.full_b,old.full_b))
            systems.append(row)
        study=read(RAW/label/'fixed'/f'f{frame}_n2_native_balanced_study.json')
        reference=study['separate_synchronized_apply_stages'];fused=study['separate_fused_synchronized_apply_stages']
        unchanged={'Z_transpose_and_first_coarse_solve','native_MAS','W_transpose_and_second_coarse_solve'}
        for r,f in zip(reference,fused):
            need(r['stage']==f['stage'],'Stage mismatch')
            stages.append(dict(scene=scene,stage=r['stage'],implementation_unchanged=r['stage'] in unchanged,
                reference_ms=r['host_ms'],fused_ms=f['host_ms'],observed_F_over_R=f['host_ms']/r['host_ms']))
    verify(inputs)
    csv_new(REPORT/'tables/cross_batch_systems.csv',systems)
    csv_new(REPORT/'tables/synchronized_stage_context.csv',stages)
    write_new(dest,dict(passed=True,source=record(Path(__file__)),inputs=inputs,systems=systems,stages=stages,
        conclusion='Use only same-snapshot paired comparisons. Single synchronized stages do not isolate causal savings; unchanged stages also vary. Repeated coefficient subtraction in fused output is additional arithmetic, but this audit does not isolate it as the slowdown cause.',
        performance_certified=False))
    print(json.dumps(dict(passed=True,systems=systems)),flush=True)
if __name__=='__main__':main()
