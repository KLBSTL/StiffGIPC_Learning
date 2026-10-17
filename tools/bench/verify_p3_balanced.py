"""Independent recount of complete single-pair P3 100-frame results."""
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read,expand
from linux_runner import require,write_new
from local_identity import verify,record,verify_active_sources
from run_p3_balanced import SESSION


def main():
    output=ROOT/'reports/innovation_p3_balanced_20261015/analysis'
    summary=read(output/'summary.json');seal=read(SESSION/'seal.json');identity=seal['identity']
    verify_active_sources(identity);verify(seal['dependencies']);verify(read(output/'input_records.json'))
    total_frames=0;total_powers=0;recount={}
    for stage in ('smoke','prefix','query','profile','whole'):
        ledger=read(SESSION/(stage+'_ledger.json'));require(ledger['complete'],'Incomplete stage')
        for row in ledger['runs']:
            t=row['task'];folder=SESSION/t['name'];q=read(folder/'requested.json');r=read(folder/'result.json')
            require(row['passed'] and r['status']=='completed' and r['cleanup_owned_job_empty'],'Invalid native run')
            require(q['exe_sha256']==identity['exe']['sha256'] and q['expanded_config']==expand(t['config']),
                'Run identity/config differs')
            powers=read(folder/'power_samples.json');total_powers+=len(powers['samples'])
            require(powers['expected']==powers['after']==seal['power_guid'] and
                powers['samples'] and all(x['guid']==seal['power_guid'] for x in powers['samples']),'Power mismatch')
            frames=read(folder/'output/stats.json')['frames']
            with (folder/'trace/frames.csv').open(newline='') as f:timings=list(csv.DictReader(f))
            require(len(frames)==len(timings)==t['config']['steps'],'Incomplete frame records')
            if stage!='whole':continue
            total_frames+=len(frames)
            require(not q['expanded_config']['diagnostics'] and q['from_zero'],'Formal run has heavy diagnostics/restore')
            require(not (folder/'cost.jsonl').exists() and not (folder/'p3_ee.jsonl').exists(),'Diagnostic output in formal run')
            p=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            require(all(x['rho_tolerance']==1e-4 and not x.get('breakdown') and not x.get('iteration_limit') for x in p),
                'PCG precision/failure')
            c=q['expanded_config'];require((c['dt'],c['ipc_cumulative_tol'],c['ipc_min_updates'])==(.04,.01,6),'Physics/stopping changed')
            actual=dict(N=len(p),P=sum(x['iterations'] for x in p),solver_ms=sum(float(x['solver_ms']) for x in timings))
            expected=next(x for pair in summary['pairs'] for x in (pair['F'],pair['W']) if x['run']==t['name'])
            require(all(math.isclose(actual[k],expected[k],rel_tol=1e-12) for k in actual),'N/P/time recount differs')
            recount[t['name']]=actual
    require(total_frames==600 and len(recount)==6,'Missing formal pair')
    for pair in summary['pairs']:
        a,b=[SESSION/pair[k]['run'] for k in ('F','W')]
        ca,cb=[read(p/'requested.json')['expanded_config'] for p in (a,b)]
        ca.pop('p3_ee_mode');cb.pop('p3_ee_mode');require(ca==cb,'F/W configurations differ beyond EE mode')
        for kind in ('state','velocity'):
            require((a/f'trace/{kind}_0000.bin').read_bytes()==(b/f'trace/{kind}_0000.bin').read_bytes(),'Initial scene/velocity differs')
        require(math.isclose(pair['solver_speedup'],recount[pair['F']['run']]['solver_ms']/recount[pair['W']['run']]['solver_ms'],rel_tol=1e-12),
            'Speedup recount differs')
    old=subprocess.check_output(['git','show','HEAD:StiffGIPC/collision/mlbvh.cu'],cwd=ROOT)
    now=(ROOT/'StiffGIPC/collision/mlbvh.cu').read_bytes();begin=b'__device__ inline void _checkEEintersection'
    require(old[old.index(begin):old.index(b'#include <collision/discrete_bvh.inl>')]==
        now[now.index(begin):now.index(b'#include <collision/p3_ee_kernels.inl>')],'Protected classifier/F/VF/FullCCD changed')
    for path in ('StiffGIPC/solver/line_search_acceptance.h','StiffGIPC/solver/ipc_options.h','StiffGIPC/solver/ipc_pcg_forcing.h'):
        require(subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT)==(ROOT/path).read_bytes(),'Stop/acceptance/forcing changed')
    native=(ROOT/'build/p3_balanced_ctest_full.log').read_text(errors='replace')
    python=(ROOT/'build/p3_balanced_python_full.log').read_text(errors='replace')
    require('100% tests passed, 0 tests failed out of 26' in native,'CTest not complete')
    require('Ran 59 tests' in python and '\nOK' in python,'Python tests not complete')
    write_new(output/'validation.json',dict(passed=True,formal_runs=6,formal_frames=total_frames,power_samples=total_powers,
        recount=recount,initial_state_and_velocity_bitwise_equal=True,protected_native_region_bytes_equal=True,
        performance_certified=False,physical_quality_certified=False,verifier=record(Path(__file__))))
    print(json.dumps(dict(passed=True,formal_frames=total_frames,power_samples=total_powers)))


if __name__=='__main__':main()
