"""Freeze a component review before choosing another implementation."""
import itertools
import json
import statistics
from config import ROOT, read, sha
from ipc_benchmark import write
from component_tuning import state_comparison
from eligibility_round import TAG


def main():
    folder=ROOT/'reports/active'
    path=folder/f'{TAG}_hang_subset_evidence.json'
    evidence=read(path)
    screen=evidence['paired_stages']['screen']
    ratios=screen['paired_speed']['hang']['off_over_on']
    assert len(ratios)==3
    repeats={}
    for arm in ('off','on'):
        repeats[arm]=[{'repeats':[a,b], 'actual_state':state_comparison(
            ROOT/f'runs/active/{TAG}_screen_hang_{arm}_{a}',
            ROOT/f'runs/active/{TAG}_screen_hang_{arm}_{b}')}
            for a,b in itertools.combinations((1,2,3),2)]
    profile={}
    for p in evidence['profiles']:
        query={}
        for k in p['query_kernels']:
            if '_selfQuery_' not in k['name']:continue
            kind=('swept_' if '_ccd<' in k['name'] else 'ordinary_')+('vf' if '_selfQuery_vf' in k['name'] else 'ee')
            query[kind]={'count':k['count'],'gpu_ms':k['sum_ms']}
        profile[p['arm']]={'cpu_frame_wall_ms':p['cpu_frame_wall_ms'],'pcg':p['target_frame_pcg'],
             'query_by_kind':query,'query_gpu_ms':p['query_gpu_sum_ms'],
             'prepare_gpu_ms':p['eligibility_gpu_sum_ms'],
             'prepare_kernel_count':sum(k['count'] for k in p['query_kernels'] if 'eligibility_' in k['name'])}
    off,on=profile['off'],profile['on']
    net=off['query_gpu_ms']-on['query_gpu_ms']-on['prepare_gpu_ms']
    ordinary_ee=off['query_by_kind']['ordinary_ee']['gpu_ms']-on['query_by_kind']['ordinary_ee']['gpu_ms']
    result={'evidence_sha256':sha(path),'paired_speed_median':statistics.median(ratios),
       'paired_elapsed_increase_median_percent':(1/statistics.median(ratios)-1)*100,
       'within_arm_actual_state_comparisons':repeats,'profiles':profile,
       'profile_accounting_only':{'query_saving_ms':off['query_gpu_ms']-on['query_gpu_ms'],
           'net_after_prepare_ms':net,'net_over_profile_off_wall_percent':net/off['cpu_frame_wall_ms']*100,
           'ordinary_ee_observed_saving_ms':ordinary_ee,
           'ordinary_ee_observed_saving_over_profile_wall_percent':ordinary_ee/off['cpu_frame_wall_ms']*100,
           'causal_gain_proven':False,'reason':'Different continuous runs, PCG 456/459 and shared GPU; launch CPU cost excluded.'},
       'candidate_disposition':'rejected_for_local_short_window_performance',
       'coverage':{'attempted':10,'completed':9,'resource_failed':1,'full_100f':0,
                   'hang_guard_complete':True,'fixed_guard_complete':False,'mixed_guard_complete':False},
       'next_direction':{'selected':'Audit matrix conversion and MAS hierarchy preparation synchronization using existing timelines.',
            'evidence_gate_only':True,'minimum_recoverable_wall_fraction':.05,
            'source_candidate':'MAS hierarchy neighborList/neighborNum synchronous D2D restoration',
            'no_kernel_implementation_authorized_by_evidence_yet':True,
            'excluded_this_round':['All-query pruning default promotion','Ordinary-EE specialization without whole-window potential',
                                    'Blind static/dynamic Hessian split','Reopening rejected graph-initialization branch']},
       'performance_certified':False,'quality_certified':False,'default_promoted':False}
    write(folder/f'{TAG}_review.json',result)
    print(json.dumps({k:result[k] for k in ('candidate_disposition','paired_speed_median','paired_elapsed_increase_median_percent','profile_accounting_only')}))


if __name__=='__main__':main()
