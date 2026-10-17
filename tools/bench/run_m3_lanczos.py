"""Append-only full-CG spectral control on the same nine M3 systems."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import csv
import json
from pathlib import Path
import time
import numpy as np
from m3_numeric import ROOT,FrozenSystem,Balanced,pcg,allow_transfer,need
from m3_lanczos import full_cg_basis
from local_identity import record,verify
from linux_runner import write_new
from m0_analyze import csv_write

PARENT=ROOT/'reports/innovation_m3_20261015/analysis_v2'
OUTPUT=ROOT/'reports/innovation_m3_20261015/full_cg_control'


def main():
    need(not OUTPUT.exists(),'Fresh full-CG control output required');OUTPUT.mkdir()
    plan=json.loads((PARENT/'ledger.json').read_text());need(plan['complete'],'Parent offline run incomplete')
    dependencies=[record(Path(__file__)),record(ROOT/'tools/bench/m3_lanczos.py'),record(ROOT/'tools/bench/m3_numeric.py')]
    write_new(OUTPUT/'plan.json',dict(parent=record(PARENT/'ledger.json'),dependencies=dependencies,
        ranks=[8,16,32],raw_lanczos_modes=96,spectral_residual_limit=.05,
        transfer_source='Previous common-tolerance solve only; full-recurrence replay is additionally charged',
        theoretical_scope='Strong finite-precision Krylov control, not certified globally optimal eigenvectors'))
    rows=[];summaries=[]
    for scene in ('Bunny','Mixed','Boxes'):
        previous=None
        paths=sorted((PARENT/scene).glob('*/result.json'),key=lambda p:(json.loads(p.read_text())['context']['frame'],json.loads(p.read_text())['context']['linear_system_id']))
        for path in paths:
            d=json.loads(path.read_text());label=d['label'];out=OUTPUT/scene/label;out.mkdir(parents=True)
            folder=ROOT/f"runs/innovation_m3_20261015/{d['kind']}_{scene.lower()}/fixed"
            prefix=folder/f"f{d['context']['frame']}_n{d['context']['inner']+1}"
            system=FrozenSystem(prefix);A,b,B0=system.A,system.b,system.apply;curves=[]
            with (path.parent/'residual_curves.csv').open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):curves.append({k:(v if k=='method' else float(v)) for k,v in r.items()})
            case_rows=[];meta={}
            def evaluate(name,Z,rank,source):
                op=Balanced(A,B0,Z[:,:rank]);r,x,_=pcg(A,b,op,B0)
                need(r['converged'],'Full-CG control PCG failed')
                np.save(out/(name+'_x.npy'),x,allow_pickle=False);r.pop('trace')
                row=dict(scene=scene,label=label,method=name,requested_rank=rank,actual_rank=op.Z.shape[1],
                    **r,source=source,setup_seconds=op.setup_seconds,Z_W_bytes=op.Z.nbytes+op.W.nbytes,
                    solution=record(out/(name+'_x.npy')))
                rows.append(row);case_rows.append(row)
            if d['kind']=='adjacent' and d['context']['inner']==1:
                allow_transfer(previous['context'],d['context']);verify([previous['basis_file']])
                z=np.load(previous['basis_file']['path'],allow_pickle=False)
                for rank in (8,16,32):
                    evaluate(f'full_transfer_ritz_{rank}',z,rank,previous['basis_file'])
                    soft=z.copy();soft[~system.soft]=0
                    evaluate(f'full_transfer_soft_ritz_{rank}',soft,rank,previous['basis_file'])
            # Freeze a practical-history spectral source separately from the
            # more expensive current-system oracle, with bitwise replay checks.
            source_trace=[r for r in curves if r['method']=='MAS_linear_reference']
            need(d['methods'][0]['residual_restarts']==0,'Restarted source recurrence cannot use ordinary Lanczos')
            x=np.load(d['methods'][0]['solution']['path'],allow_pickle=False)
            start=time.perf_counter();z,info=full_cg_basis(A,b,B0,source_trace,x);info['construction_seconds']=time.perf_counter()-start
            np.save(out/'source_ritz.npy',z,allow_pickle=False);meta['common_source']=info
            source_packet=dict(context=d['context'],basis_file=record(out/'source_ritz.npy'))
            need(d['reference']['residual_restarts']==0,'Restarted oracle recurrence cannot use ordinary Lanczos')
            ref_trace=[r for r in curves if r['method']=='reference_1e_8']
            x=np.load(d['reference']['solution']['path'],allow_pickle=False)
            start=time.perf_counter();z,info=full_cg_basis(A,b,B0,ref_trace,x);info['construction_seconds']=time.perf_counter()-start
            meta['current_oracle']=info
            for rank in (8,16,32):
                evaluate(f'full_oracle_ritz_{rank}',z,rank,None)
                soft=z.copy();soft[~system.soft]=0
                evaluate(f'full_oracle_soft_ritz_{rank}',soft,rank,None)
            write_new(out/'result.json',dict(scene=scene,label=label,parent=record(path),basis_diagnostics=meta,
                source_packet=source_packet,methods=case_rows,performance_certified=False))
            summaries.append(record(out/'result.json'))
            print(json.dumps(dict(scene=scene,label=label,baseline=d['methods'][0]['iterations'],
                source_rank=meta['common_source']['independent_rank'],oracle_rank=meta['current_oracle']['independent_rank'],
                best=min((r['iterations'],r['method']) for r in case_rows))),flush=True)
            if d['kind']=='adjacent' and d['context']['inner']==0:previous=source_packet
    verify(dependencies);write_new(OUTPUT/'summary.json',dict(complete=True,cases=summaries,methods=len(rows),
        all_18_recurrence_replays_bitwise_equal=True,performance_certified=False))


if __name__=='__main__':main()
