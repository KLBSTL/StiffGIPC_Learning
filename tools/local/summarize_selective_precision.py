"""Summarize finite precision evidence and separate arithmetic from tolerances."""
from pathlib import Path
import json
import os
import sqlite3
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):sys.path.insert(0,str(ROOT/'tools'/folder))
from config import read
from linux_runner import require,write_new
from local_identity import record,verify
from audit_recycling_reference import preconditioner
from cpu_fixed_reference import load_snapshot
from screen_pcg_tolerance import solve
import numpy as np


def main():
    out=ROOT/'runs/selective_precision_summary_20261009'
    require(not out.exists(),'Fresh summary required');out.mkdir()
    (out/'controller_at_run.py').write_bytes(Path(__file__).read_bytes())
    cpu_path=ROOT/'runs/selective_precision_cpu_20261009/analysis.json';cpu=read(cpu_path)
    tolerance=[]
    for system in cpu['systems']:
        prefix=Path(system['prefix']);matrix,rhs,active,_,_,_=load_snapshot(prefix)
        require(active.all(),'Zero DOFs require explicit treatment')
        apply,_=preconditioner(prefix,read(Path(str(prefix)+'_meta.json')))
        cases=[];baseline=None
        for eta in (.01,np.sqrt(1e-5),.001):
            x,result=solve(matrix,rhs,apply,eta)
            if baseline is None:baseline=x
            result['relative_direction_difference']=float(np.linalg.norm(x-baseline)/np.linalg.norm(baseline))
            cases.append(result)
        tolerance.append(dict(prefix=str(prefix),cases=cases))
        print(dict(system=prefix.name,tolerance_iterations=[r['iterations'] for r in cases]),flush=True)
    trace_folder=ROOT/'runs/selective_precision_fixture_trace_20261009'
    result=read(trace_folder/'result.json');require(result['passed'] and result['cleanup_owned_job_empty'],'Native fixture incomplete')
    verify([result['trace'],result['fixture']])
    sqlite_path=trace_folder/'trace_export.sqlite';require(sqlite_path.is_file(),'Export unavailable')
    connection=sqlite3.connect('file:'+sqlite_path.as_posix()+'?mode=ro',uri=True)
    rows=list(connection.execute('SELECT k.start,k.end,s.value,k.registersPerThread,k.staticSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start'))
    memsets=list(connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start'))
    factor=[r for r in rows if '__inverse6_' in r[2]]
    require(len(factor)==1 and len(rows)==487,'Unexpected native trace coverage')
    kernels=[r for r in rows if '__inverse6_' not in r[2]]
    require(len(kernels)==162*3,'Unexpected number of actions')
    actions=[];previous=factor[0][1]
    for index in range(162):
        triplet=kernels[3*index:3*index+3]
        require('buildMultiLevelR' in triplet[0][2] and 'schwarzLocalXSym6' in triplet[1][2] and 'collectFinalZ' in triplet[2][2],'Stage sequence differs')
        local=triplet[1][2]
        mode='legacy' if '_wide' not in local else 'wide' if 'MasMatrixSymf' in local else 'inverse64'
        zero=[(end-start)/1000 for start,end in memsets if previous<=start and end<=triplet[0][0]]
        require(len(zero)==2,'Workspace zero coverage differs')
        values=[(r[1]-r[0])/1000 for r in triplet]
        actions.append(dict(mode=mode,restrict_us=values[0],local_us=values[1],prolong_us=values[2],
            workspace_zero_us=sum(zero),GPU_work_us=sum(values)+sum(zero)))
        previous=triplet[2][1]
    expected=['legacy']*40+['wide']*40+['inverse64']*40+['legacy']*40+['inverse64']*2
    require([r['mode'] for r in actions]==expected,'Phase order differs')
    costs={}
    for mode in ('legacy','wide','inverse64'):
        selected=[r for r in actions[:160] if r['mode']==mode]
        costs[mode]={key:float(np.mean([r[key] for r in selected])) for key in
            ('restrict_us','local_us','prolong_us','workspace_zero_us','GPU_work_us')}
        costs[mode]['observations']=len(selected)
        costs[mode]['GPU_work_quantiles_us']=np.quantile([r['GPU_work_us'] for r in selected],[0,.5,.9,1]).tolist()
    for mode in ('wide','inverse64'):
        costs[mode]['M_only_break_even_iteration_ratio']=costs['legacy']['GPU_work_us']/costs[mode]['GPU_work_us']
    native_ledger=read(ROOT/'runs/selective_precision_native_20261009/ledger.json')
    source_before=read(ROOT/'runs/selective_precision_native_20261009/identity.json')['native_source_before']
    verify(source_before)
    report=dict(controller=record(__file__),CPU_evidence=record(cpu_path),CPU_cases=cpu['systems'],
        tolerance_cases=tolerance,tolerance_helper=record(ROOT/'tools/local/screen_pcg_tolerance.py'),
        native_fixture_evidence=record(trace_folder/'result.json'),fixture=result['fixture'],trace=result['trace'],
        sqlite_evidence=record(sqlite_path),native_kernel_costs=costs,trace_kernel_events=len(rows),
        factor_kernel_us=(factor[0][1]-factor[0][0])/1000,
        full_scene_attempt_evidence=record(ROOT/'runs/selective_precision_native_20261009/ledger.json'),
        full_scene_status=[dict(name=r['task']['name'],status=r['result']['status'],frames=r['result']['recorded_frames']) for r in native_ledger['runs']],
        native_source_unchanged=True,
        cautions=['CPU stage/reduction order differs from CUDA. Same original MAS norm for arithmetic screens.',
            'Kernel cost includes workspace memset GPU work, excludes launch gaps, PCG, SpMV, preparation and allocations. Instrumented shared-WDDM diagnosis; not native solver wall time.',
            'Inverse64 uses an existing unoptimized factor kernel; measured one-time factor cost has no matching legacy preparation comparison.',
            'M-only iteration threshold is for the MAS GPU-work component, not whole-PCG or whole-scene break-even.',
            'Partial memory-budget-stopped full-scene attempt is not a performance or quality result. User selected fixed-system analysis.'],
        physical_quality_certified=False,whole_scene_performance_certified=False)
    write_new(out/'analysis.json',report)
    write_new(ROOT/'reports/SELECTIVE_PRECISION_RESULTS.json',report)
    print(json.dumps(dict(costs=costs,native_source_unchanged=True),indent=2))


if __name__=='__main__':main()
