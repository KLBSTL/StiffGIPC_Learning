"""Frozen-trace schedule screening and cost budgets; no solver/GPU changes.

Never invent a weighted residual that the old execution did not record, and
never force a check at its old stopping point just to complete a replay.
"""
import collections
import json
import math
from pathlib import Path
import statistics as st
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_native_balanced import csv_new
RAW=ROOT/'runs/innovation_p5_mas_cached_20261017'
OUT=ROOT/'reports/innovation_p5_next_steps_20261017'
PARENT=ROOT/'reports/innovation_p5_mas_cached_20261017/DELIVERY_SEAL.json'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def replay(result,family,period,limit=2000):
    require(family in ['periodic','weighted_cooldown'] and period in [2,4,8],'Unknown policy')
    require(result['converged'] and len(result['trace'])==result['iterations'],'Incomplete source trace')
    last=0;active=False;split=False;checks=[];screens=0
    for point in result['trace']:
        i=point['iteration'];active=active or point['pre_update_rho']<=1e-4*result['rho_initial']
        override=result.get('zero_rho_true_quality_checked',False) and i==result['iterations']
        due=False
        if active:
            if family=='periodic':due=last==0 or i-last>=period or i==limit
            elif split:due=i-last>=period or i==limit
            else:
                value=point['recursive_relative_l2']
                require(value is not None and math.isfinite(value) and value>=0,'Missing recursive screen')
                screens+=1;due=last==0 or i-last>=8 or value<=.012 or i==limit
        due=due or override
        if not due:continue
        if not point['quality_checked']:
            return dict(status='unobserved_inside_trace',iteration=i,checks=len(checks),screens=screens,accepted_iteration=None)
        l2=point['true_relative_l2'];weighted=point['common_native_MAS_relative']
        require(math.isfinite(l2) and math.isfinite(weighted) and min(l2,weighted)>=0,'Invalid true quality')
        checks.append(i);last=i
        if l2<=.0095 and weighted<=.0095:
            return dict(status='observed_stop',iteration=i,checks=len(checks),screens=screens,accepted_iteration=i)
        split=l2<=.0095 and weighted>.0095
    # The original terminal iteration is not the candidate's iteration limit.
    # Without an observed success, all subsequent quality remains unknown.
    next_due=min(last+period,limit) if family=='periodic' or split else None
    return dict(status='right_censored',iteration=result['iterations'],checks=len(checks),screens=screens,
        accepted_iteration=None,next_scheduled_opportunity=next_due)

def contracts():
    def trace(n,pass_at,missing=()):
        rows=[]
        for i in range(1,n+1):
            checked=i not in missing
            rows.append(dict(iteration=i,pre_update_rho=1e-5,recursive_relative_l2=.005,quality_checked=checked,
                true_relative_l2=.005 if checked else None,common_native_MAS_relative=(.009 if i>=pass_at else .02) if checked else None))
        return dict(converged=True,iterations=n,rho_initial=1.,trace=rows)
    x=replay(trace(5,5),'periodic',2);require(x['status']=='observed_stop' and x['accepted_iteration']==5 and x['checks']==3,'Aligned success')
    x=replay(trace(4,4),'periodic',2);require(x['status']=='right_censored' and x['next_scheduled_opportunity']==5,'Do not force old terminal check')
    x=replay(trace(5,5,missing=(3,)),'periodic',2);require(x['status']=='unobserved_inside_trace' and x['iteration']==3,'Do not invent missing residual')
    x=replay(trace(5,5),'weighted_cooldown',2);require(x['status']=='observed_stop' and x['screens']==1 and x['checks']==3,'Cooldown removes repeated norm screening')
    # A true residual need not decrease monotonically. L2 regression exits the
    # special phase, and later acceptance still needs both explicit metrics.
    t=trace(5,5);t['trace'][2]['true_relative_l2']=.02
    x=replay(t,'weighted_cooldown',2)
    require(x['status']=='right_censored' and x['screens']==2 and x['next_scheduled_opportunity']==6,
        'L2 regression exits cooldown and changes the next schedule; do not force old stop')
    t=trace(4,4);t['zero_rho_true_quality_checked']=True
    x=replay(t,'periodic',2);require(x['status']=='observed_stop' and x['accepted_iteration']==4,'Preserve zero-rho explicit override')
    x=replay(trace(4,4),'periodic',2,limit=4);require(x['status']=='observed_stop','Actual budget forces explicit check')
    return dict(passed=True,meaningful_cases=7)

def main():
    require(not OUT.exists(),'Fresh offline analysis output required');verify(read(PARENT)['files']);tests=contracts();OUT.mkdir()
    packet=read(RAW/'measure/data.json');details=[];budgets=[]
    for c in packet['cases']:
        for execution in ['host','graph_apply1']:
            banks={method:[r for r in c['runs'] if r['comparison']=='AB' and r['execution']==execution and r['method']==method] for method in ['MAS','cached']}
            require(all(len(v)==24 for v in banks.values()),'Exact source arm count')
            mas=st.mean(r['solve_host_ms'] for r in banks['MAS']);bank=banks['cached'];time=st.mean(r['solve_host_ms'] for r in bank)
            quality=st.mean(r['result']['quality_host_ms'] for r in bank);screen=st.mean(r['result']['screen_host_ms'] for r in bank)
            require(time>quality+screen>=0,'Invalid inclusive budget')
            budgets.append(dict(scene=c['name'],execution=execution,MAS_mean_ms=mas,cached_mean_ms=time,quality_mean_ms=quality,screen_mean_ms=screen,
                optimistic_no_quality_ratio=mas/(time-quality),optimistic_no_quality_or_screen_ratio=mas/(time-quality-screen),
                kind='Arithmetic optimistic model with all other work frozen; not measured or a hardware-independent bound'))
            for method,bank in banks.items():
                for index,row in enumerate(bank):
                    result=row['result']
                    for family in ['periodic','weighted_cooldown']:
                        for period in [2,4,8]:
                            value=replay(result,family,period);model=None
                            if value['status']=='observed_stop':
                                require(value['accepted_iteration']==result['iterations'],'A fixed observed trace cannot stop before its first accepted check')
                                q=result['quality_checks'];s=result['screen_checks'];require(q>0 and s>0,'Invalid normalization')
                                model=row['solve_host_ms']-result['quality_host_ms']-result['screen_host_ms']
                                model+=value['checks']*result['quality_host_ms']/q+value['screens']*result['screen_host_ms']/s
                            details.append(dict(scene=c['name'],execution=execution,method=method,source_index=index,family=family,period=period,
                                original_iterations=result['iterations'],original_checks=result['quality_checks'],original_screens=result['screen_checks'],
                                **value,conditional_uniform_check_cost_model_ms=model))
    grouped=collections.defaultdict(list)
    for row in details:grouped[tuple(row[k] for k in ['scene','execution','method','family','period'])].append(row)
    summary=[]
    for key,bank in sorted(grouped.items()):
        counts=collections.Counter(r['status'] for r in bank);known=[r for r in bank if r['status']=='observed_stop']
        summary.append(dict(scene=key[0],execution=key[1],method=key[2],family=key[3],period=key[4],samples=len(bank),
            observed_stop=counts['observed_stop'],unobserved_inside_trace=counts['unobserved_inside_trace'],right_censored=counts['right_censored'],
            known_subset_mean_checks=st.mean(r['checks'] for r in known) if known else None,
            known_subset_mean_screens=st.mean(r['screens'] for r in known) if known else None,
            known_subset_mean_model_ms=st.mean(r['conditional_uniform_check_cost_model_ms'] for r in known) if known else None))
    csv_new(OUT/'schedule_samples.csv',details);csv_new(OUT/'schedule_summary.csv',summary);csv_new(OUT/'optimistic_cost_budgets.csv',budgets)
    write_new(OUT/'ANALYSIS.json',dict(parent=record(PARENT),source=record(Path(__file__)),contracts=tests,samples=len(details),cells=summary,budgets=budgets,
        GPU_runs=0,solver_modified=False,performance_certified=False,missing_residuals_fabricated=False))
    sources=OUT/'sources';sources.mkdir();(sources/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_new(OUT/'SOURCE_SEAL.json',dict(files=tree(OUT),source=record(Path(__file__))))
    print(json.dumps(dict(contracts=tests,counterfactual_rows=len(details),GPU_runs=0,
        Mixed_cached=[r for r in summary if r['scene']=='mixed_rhs' and r['method']=='cached'],budgets=budgets),indent=2),flush=True)
if __name__=='__main__':main()
