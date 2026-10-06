"""Protected local-factor comparisons; preparation cost must be reported separately."""
import argparse
import json
import statistics
from config import ROOT,read,sha

def summarize(path):
    study=read(path);groups=[]
    assert study['system_unchanged'] and study['primary_restored_bitwise']
    for mode in ('host','graph'):
        rows=[r for r in study['factor_action_pairs'] if r['mode']==mode]
        ratios=[]
        for pair in (1,2,3):
            old=next(r for r in rows if r['pair']==pair and r['factor_action']=='triangular')
            new=next(r for r in rows if r['pair']==pair and r['factor_action']=='factor_inverse')
            ratios.append(old['solve_ms']/new['solve_ms'])
        groups.append({'mode':mode,'paired_solve_ratios':ratios,'median_solve_ratio':statistics.median(ratios),
            'arms':{arm:{'iterations':[r['iterations'] for r in rows if r['factor_action']==arm],
                         'median_solve_ms':statistics.median(r['solve_ms'] for r in rows if r['factor_action']==arm),
                         'max_difference_to_first_triangular':max(r['triangular_difference']['relative'] for r in rows if r['factor_action']==arm),
                         'true_relative_residual_range':[min(r['true_relative_residual'] for r in rows if r['factor_action']==arm),
                                                         max(r['true_relative_residual'] for r in rows if r['factor_action']==arm)]}
                    for arm in ('triangular','factor_inverse')}})
    return {'study':str(path),'sha256':sha(path),'frame':study['frame'],'direction':study['direction'],
            'system_unchanged':True,'main_solution_restored':True,'comparisons':groups}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directories',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    rows=[summarize(path) for folder in a.directories for path in sorted((ROOT/folder).glob('*_study.json'))]
    assert rows
    result={'scope':'Same A/b/L/CSR default-rho PCG only; extra B preparation excluded here and required in total-linear comparison',
            'physical_certified':False,'performance_certified':False,'systems':rows}
    with (ROOT/a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'systems':[{'frame':r['frame'],'direction':r['direction'],
          'ratios':{c['mode']:c['median_solve_ratio'] for c in r['comparisons']}} for r in rows]}))
