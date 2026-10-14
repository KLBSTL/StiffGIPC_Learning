"""Independent final raw-run, physics-source and checkpoint delivery checks."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,verify_active_sources
from linux_runner import require,write_new
from config import read

def main():
    session=ROOT/'runs/innovation_p2a_verified_20261014'
    output=ROOT/'reports/innovation_p2a_20261014/analysis_v2/data_validation.json'
    require(not output.exists(),'Final validation is append-only')
    seal=read(session/'seal.json');identity=seal['identity']
    frozen=read(ROOT/'build/innovation_p2a_native_snapshot_20261014/snapshot_identity.json')
    verify_active_sources(frozen)
    verify(identity['sources']+identity['objects']+identity['build_evidence']+[identity['exe']]+identity['dlls'])
    verify([seal['controller'],seal['runner']])
    resume=read(session/'resume_inputs.json');verify(resume['inherited']+[resume['controller']])
    for pair in resume['copies']:
        verify([pair['original'],pair['copy']]);require(pair['original']['sha256']==pair['copy']['sha256'],'Copied input altered')
    archived=ROOT/'reports/innovation_p2a_20261014/run_innovation_p2a_before_path_fix.py'
    for prior in ('innovation_p2a_20261014','innovation_p2a_bunny_20261014','innovation_p2a_full_retry_20261014'):
        require(read(ROOT/'runs'/prior/'seal.json')['controller']['sha256']==record(archived)['sha256'],
            'Pre-fix historical controller not preserved')
    summary=read(output.parent/'summary.json');rows=[];restores=0;audits=0;powers=0;tree_passes=0;frames_total=0
    for stage in ('smoke','prefix','cost','audit','matched','whole'):
        ledger=read(session/(stage+'_ledger.json'));require(ledger['complete'],'Incomplete stage')
        origin=Path(ledger['inherited_from']['path']).parent if 'inherited_from' in ledger else session
        if 'inherited_from' in ledger:verify([ledger['inherited_from']])
        for entry in ledger['runs']:
            task=entry['task'];folder=origin/task['name'];r=read(folder/'result.json');q=read(folder/'requested.json')
            require(r['status']=='completed' and r['cleanup_owned_job_empty'] and r['exit_code']==0,'Invalid completed run')
            require(q['exe_sha256']==identity['exe']['sha256'],'Unexpected program')
            cfg=q['expanded_config'];resolved=read(folder/'resolved_config.json')
            require(read(folder/'config_validation.json')['passed'],'Invalid resolved configuration')
            require((cfg['dt'],cfg['pcg_rho_tol'],cfg['ipc_cumulative_tol'],cfg['ipc_min_updates'])==(.04,1e-4,.01,6),
                'Tolerance/time step changed')
            for key in ('lazy_scene','merge_abd_energy','audit'):
                require(resolved['p2_execution'][key]==cfg['p2_'+key],'P2 resolution differs')
                require(q['environment']['GIPC_P2_'+{'lazy_scene':'LAZY_SCENE','merge_abd_energy':'MERGE_ABD_ENERGY','audit':'AUDIT'}[key]]
                    ==str(int(cfg['p2_'+key])),'P2 native environment differs')
            if stage in ('matched','whole'):
                require(cfg['diagnostics']==[] and not cfg['p2_audit'] and not cfg['discrete_bvh_validate'],
                    'Diagnostic inserted into paired timing')
                require(cfg['ipc_termination']=='legacy' and q['environment']['GIPC_IPC_PCG_FORCING']=='off','Stopping/forcing differs')
            frames=read(folder/'output/stats.json')['frames'];require(len(frames)==cfg['steps'],'Missing stats frame')
            require(not any(f.get('newton_exit') in (None,'iteration_limit') for f in frames),'Invalid nonlinear exit')
            pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
            require(not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg),'Linear failure')
            with (folder/'trace/frames.csv').open(encoding='utf-8-sig',newline='') as stream:timings=list(csv.DictReader(stream))
            require([int(x['frame']) for x in timings]==list(range(1,cfg['steps']+1)),'Frame order differs')
            measured=sum(float(x['solver_ms']) for x in timings);observation=entry['observation']
            require(len(pcg)==observation['N'] and sum(p['iterations'] for p in pcg)==observation['P'],'N/P recount differs')
            require(math.isclose(measured,observation['solver_ms'],rel_tol=1e-12),'Raw CPU solver timer differs')
            for local in range(cfg['steps']+1):
                for name in ('state','velocity'):
                    path=folder/f'trace/{name}_{local:04d}.bin'
                    require(path.is_file() and path.stat().st_size>0,'Missing actual state/velocity')
            state_audits=[a for f in frames for a in f.get('p2_abd_energy_audit',[])]
            require(all(a['bitwise_equal'] for a in state_audits),'Component mismatch');audits+=len(state_audits)
            power=read(folder/'power_samples.json');require(power['expected']==seal['power_guid'] and power['after']==seal['power_guid']
                and all(s['guid']==seal['power_guid'] for s in power['samples']),'Power mode differs')
            powers+=len(power['samples']);frames_total+=cfg['steps']
            if 'checkpoint_from' in task:
                cp=session/task['checkpoint_from'];meta=read(cp/'checkpoint.json');native=read(folder/'output/checkpoint_restore.json')
                require(q['from_zero'] is False and Path(q['checkpoint_load']).is_absolute() and Path(q['checkpoint_load'])==cp,
                    'Restore origin/path incorrect')
                require(meta['binary_id']==identity['exe']['sha256'] and meta['next_physical_frame']==task['target_frame'],
                    'Checkpoint identity/frame mismatch')
                require(native['device_bytes_verified'] and native['context_verified'] and native['restored_buffers']==40==len(meta['buffers']),
                    'Native checkpoint validation missing')
                for member,trace in [('vertices','state'),('velocities','velocity')]:
                    require((cp/(member+'.bin')).read_bytes()==(folder/f'trace/{trace}_0000.bin').read_bytes(),'Restored initial data differs')
                restores+=1
            if stage=='audit':
                for f in frames:
                    for kind in ('face','edge'):
                        tree=f['discrete_bvh'][kind]
                        require(tree['validation_failed']==0 and tree['validation_passed']==tree['validation_calls'],'Tree multiset failure')
                        tree_passes+=tree['validation_passed']
            rows.append(dict(stage=stage,name=task['name'],scene=task['scene'],arm=task['arm'],repeat=task.get('repeat',0),
                solver_ms=measured,line_search_ms=sum(n.get('phase_ms',{}).get('line_search',0) for f in frames for n in f['newton'])))
    for stage,groups in summary['paired'].items():
        for group,metrics in groups.items():
            scene,arm=group.split('/')
            for metric,values in metrics.items():
                ratios=[]
                for repeat in (1,2,3):
                    control=next(x for x in rows if (x['stage'],x['scene'],x['arm'],x['repeat'])==(stage,scene,'control',repeat))
                    candidate=next(x for x in rows if (x['stage'],x['scene'],x['arm'],x['repeat'])==(stage,scene,arm,repeat))
                    ratios.append(control[metric]/candidate[metric])
                recomputed=math.exp(sum(math.log(x) for x in ratios)/3)
                require(math.isclose(recomputed,values['geometric_mean'],rel_tol=1e-12),'Paired ratio differs')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
    require(not git('diff','HEAD','--name-only','--','baseline','StiffGIPC/fem','StiffGIPC/linear_system','StiffGIPC/cuda_tools'),
        'Frozen baseline, FEM/PCG or cuda helpers changed')
    for path in ('StiffGIPC/solver/line_search_acceptance.h','StiffGIPC/solver/ipc_options.h','StiffGIPC/solver/ipc_pcg_forcing.h'):
        require(git('show','HEAD:'+path)==(ROOT/path).read_text(encoding='utf-8'),'Acceptance/stop/forcing source changed')
    path='StiffGIPC/abd_system/abd_system_function/cal_abd_energy.cu'
    original=git('show','HEAD:'+path);current=(ROOT/path).read_text(encoding='utf-8')
    require(original[original.index('__global__'):original.index('Float ABDSystem::cal_abd_kinetic_energy')]
        ==current[current.index('__global__'):current.index('void ABDSystem::enqueue_kinetic_energy')],
        'ABD mathematical kernels changed')
    path='StiffGIPC/core/GIPC.cu';original=git('show','HEAD:'+path);current=(ROOT/path).read_text(encoding='utf-8')
    pattern=r'bool GIPC::lineSearch\([\s\S]*?^}'
    before=re.search(pattern,original,re.M).group();after=re.search(pattern,current,re.M).group()
    require(before.replace('    wait_device();','    { gipc::CostScope cost_wait("ipc.line_search.entry_wait",false); wait_device(); }')==after,
        'Line-search semantics changed beyond observational wait scope')
    require((len(rows),frames_total,restores,audits,tree_passes)==(79,384,54,77,140),'Final count differs')
    require(summary['diagnostic_only'] and not summary['formal_performance_certified'] and not summary['physical_quality_certified'],
        'Improper certification claim')
    ctest=ROOT/'build/innovation_p2a_20261014/p2_ctest_full.log'
    require('100% tests passed, 0 tests failed out of 24' in ctest.read_text(errors='replace'),'Full native CTest failed')
    write_new(output,dict(passed=True,verified_runs=len(rows),verified_frames=frames_total,restores=restores,
        buffers_per_restore=40,abd_bitwise_pairs=audits,tree_validations=tree_passes,power_samples=powers,
        independent_N_P_frame_time_and_ratio_recount=True,physics_kernels_and_line_search_source_verified=True,
        inherited_checkpoint_bytes_unchanged=True,summary=record(output.parent/'summary.json'),
        native_tests=record(ctest),verifier=record(Path(__file__)),formal_performance_certified=False,physical_quality_certified=False))
    print(json.dumps(dict(passed=True,runs=len(rows),frames=frames_total,restores=restores,power_samples=powers)))

if __name__=='__main__':main()
