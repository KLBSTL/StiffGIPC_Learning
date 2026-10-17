"""Independent recount and immutable build/source checks for the finite P3 study."""
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read
from linux_runner import require,write_new
from local_identity import record,verify,verify_active_sources


def main():
    session=ROOT/'runs/innovation_p3_20261015'
    output=ROOT/'reports/innovation_p3_20261015/analysis/data_validation.json'
    require(not output.exists(),'Append-only verification')
    seal=read(session/'seal.json');identity=seal['identity']
    verify_active_sources(identity)
    verify(identity['sources']+identity['objects']+identity['build_evidence']+[identity['exe']]+identity['dlls'])
    verify([seal['controller'],seal['runner']])
    control=read(session/'control_query_plan.json');verify([control['controller'],control['original_seal']])
    summary=read(output.parent/'summary.json');rows=[];restores=0;powers=0;frames_total=0
    default_fallbacks=0;tiny_fallbacks=0;max_raw=0;query_pairs=0;typed_checks=0
    for stage in ('smoke','prefix','query','matched','fallback','whole','control_query'):
        ledger=read(session/(stage+'_ledger.json'));require(ledger['complete'],'Incomplete stage')
        for entry in ledger['runs']:
            task=entry['task'];folder=session/task['name'];r=read(folder/'result.json');q=read(folder/'requested.json')
            require(entry['passed'] and r['status']=='completed' and r['exit_code']==0 and r['cleanup_owned_job_empty'],'Run failed')
            require(q['exe_sha256']==identity['exe']['sha256'],'Wrong binary')
            cfg=q['expanded_config'];resolved=read(folder/'resolved_config.json')
            require(read(folder/'config_validation.json')['passed'],'Resolved config failed')
            require((cfg['dt'],cfg['pcg_rho_tol'],cfg['ipc_cumulative_tol'],cfg['ipc_min_updates'])==(.04,1e-4,.01,6),'Numerical settings changed')
            require(all(not cfg['p2_'+k] and not resolved['p2_execution'][k] for k in ('lazy_scene','merge_abd_energy','audit')),'P2 enabled')
            require(not cfg['contact_pool'] and resolved['p3_ee']['mode']==task['mode'],'Wrong EE dispatch')
            require(q['environment']['GIPC_P3_EE_MODE']==task['mode'] and int(q['environment']['GIPC_P3_EE_QUEUE_CAPACITY'])==cfg['p3_ee_queue_capacity'],'Native environment differs')
            if stage in ('matched','whole'):
                require(cfg['diagnostics']==[] and not cfg['discrete_bvh_validate'] and cfg['ipc_termination']=='legacy','Instrumented timing')
            require(q['environment']['GIPC_IPC_PCG_FORCING']=='off','Forcing changed')
            frames=read(folder/'output/stats.json')['frames'];require(len(frames)==cfg['steps'],'Frame missing')
            require(all(f.get('newton_exit') not in (None,'iteration_limit') for f in frames),'Nonlinear failure')
            pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            require(not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg),'Linear failure')
            with (folder/'trace/frames.csv').open(encoding='utf-8-sig',newline='') as stream:timings=list(csv.DictReader(stream))
            require([int(x['frame']) for x in timings]==list(range(1,cfg['steps']+1)),'Trace order differs')
            measured=sum(float(x['solver_ms']) for x in timings);o=entry['observation']
            require(len(pcg)==o['N'] and sum(p['iterations'] for p in pcg)==o['P'],'N/P recount differs')
            require(math.isclose(measured,o['solver_ms'],rel_tol=1e-12),'CPU solver time recount differs')
            for f in frames:
                v=f['p3_ee'];require(v['mode']==task['mode'],'Runtime mode differs')
                require(v['w_queries']>0 if task['mode']=='two_stage' else v['w_queries']==0,'Runtime dispatch not used')
                max_raw=max(max_raw,v['raw_peak'])
                if stage=='fallback':tiny_fallbacks+=v['complete_fused_fallbacks']
                else:default_fallbacks+=v['complete_fused_fallbacks']
            for local in range(cfg['steps']+1):
                for name in ('state','velocity'):
                    path=folder/f'trace/{name}_{local:04d}.bin'
                    require(path.is_file() and path.stat().st_size>0,'Missing real state/velocity')
            power=read(folder/'power_samples.json')
            require(power['samples'] and power['expected']==power['after']==seal['power_guid'] and all(s['guid']==seal['power_guid'] for s in power['samples']),'Power changed')
            powers+=len(power['samples']);frames_total+=cfg['steps']
            if 'checkpoint_from' in task:
                cp=session/task['checkpoint_from'];meta=read(cp/'checkpoint.json');native=read(folder/'output/checkpoint_restore.json')
                require(not q['from_zero'] and Path(q['checkpoint_load']).is_absolute() and Path(q['checkpoint_load'])==cp,'Wrong restore origin')
                require(meta['binary_id']==identity['exe']['sha256'] and meta['next_physical_frame']==task['target_frame'],'Wrong checkpoint')
                require(native['device_bytes_verified'] and native['context_verified'] and native['restored_buffers']==40==len(meta['buffers']),'Restore incomplete')
                for member,trace in [('vertices','state'),('velocities','velocity')]:
                    require((cp/(member+'.bin')).read_bytes()==(folder/f'trace/{trace}_0000.bin').read_bytes(),'Initial restore differs')
                restores+=1
            if stage in ('query','control_query'):
                study=json.loads((folder/'p3_ee.jsonl').read_text())
                require(study['passed'] and study['raw_count_matches_original_narrow_calls'],'Query incomplete')
                require(len(study['pairs'])==7,'Missing query pairs');query_pairs+=7
                for pair in study['pairs']:
                    for arm in ('F','W'):
                        require(pair[arm]['counts_equal'] and pair[arm]['typed_multiset_equal'],'Typed multiset failed');typed_checks+=1
                snap=list((folder/'fixed').glob('*_study.json'));require(len(snap)==1,'Missing matrix')
                v=read(snap[0]);require(v['system_unchanged'] and v['primary_restored_bitwise'],'Fixed study mutated primary')
            rows.append(dict(stage=stage,scene=task['scene'],mode=task['mode'],repeat=task.get('repeat',0),
                solver_ms=measured,line_search_ms=sum(n['phase_ms'].get('line_search',0) for f in frames for n in f['newton'])))
    for stage,groups in summary['paired'].items():
        for scene,metrics in groups.items():
            for metric,value in metrics.items():
                def find(mode,r):return next(x for x in rows if (x['stage'],x['scene'],x['mode'],x['repeat'])==(stage,scene,mode,r))[metric]
                ratios=[find('fused',r)/find('two_stage',r) for r in (1,2,3)]
                require(ratios==value['pairs'] and math.isclose(math.exp(sum(map(math.log,ratios))/3),value['geometric_mean'],rel_tol=1e-12),'Paired ratio recount differs')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
    require(not git('diff','HEAD','--name-only','--','baseline','Assets','StiffGIPC/fem','StiffGIPC/linear_system','StiffGIPC/cuda_tools','StiffGIPC/abd_system'),'Protected sources changed')
    for path in ('StiffGIPC/solver/line_search_acceptance.h','StiffGIPC/solver/ipc_options.h','StiffGIPC/solver/ipc_pcg_forcing.h'):
        require(git('show','HEAD:'+path)==(ROOT/path).read_text(encoding='utf-8'),'Acceptance/stop/forcing changed')
    path='StiffGIPC/core/GIPC.cu';old=git('show','HEAD:'+path);now=(ROOT/path).read_text(encoding='utf-8')
    pattern=r'bool GIPC::lineSearch\([\s\S]*?^}'
    require(re.search(pattern,old,re.M).group()==re.search(pattern,now,re.M).group(),'Line search changed')
    path='StiffGIPC/collision/mlbvh.cu';old=git('show','HEAD:'+path);now=(ROOT/path).read_text(encoding='utf-8')
    begin='__device__ inline void _checkEEintersection'
    require(old[old.index(begin):old.index('#include <collision/discrete_bvh.inl>')]==now[now.index(begin):now.index('#include <collision/p3_ee_kernels.inl>')],'Original F/VF/FullCCD/classifier changed')
    ctest=ROOT/'build/p3_ctest_full.log';python=ROOT/'build/p3_python_full.log'
    require('100% tests passed, 0 tests failed out of 26' in ctest.read_text(errors='replace'),'Native regression failed')
    require('Ran 50 tests' in python.read_text(errors='replace') and '\nOK' in python.read_text(errors='replace'),'Python regression failed')
    require((len(rows),frames_total,restores,query_pairs,typed_checks)==(53,356,30,63,126),'Final counts differ')
    require(default_fallbacks==0 and tiny_fallbacks>0,'Unexpected queue behavior')
    require(not summary['performance_certified'] and not summary['physical_quality_certified'],'Improper certification')
    write_new(output,dict(passed=True,runs=len(rows),frames=frames_total,restores=restores,buffers_per_restore=40,
        exact_input_query_pairs=query_pairs,timed_typed_multiset_checks=typed_checks,power_samples=powers,
        normal_capacity_fallbacks=default_fallbacks,capacity_one_fallbacks=tiny_fallbacks,raw_peak=max_raw,
        N_P_frame_time_and_ratio_independently_recounted=True,protected_kernels_and_line_search_unchanged=True,
        native_tests=record(ctest),python_tests=record(python),verifier=record(Path(__file__)),summary=record(output.parent/'summary.json')))
    print(json.dumps(dict(passed=True,runs=len(rows),frames=frames_total,restores=restores,query_pairs=query_pairs,
        normal_fallbacks=default_fallbacks,tiny_fallbacks=tiny_fallbacks,raw_peak=max_raw,power_samples=powers)))


if __name__=='__main__':main()
