"""Summarize protected, same-system restriction comparisons without certifying a trajectory."""
import argparse
import json
import statistics
from config import ROOT,read,sha

def summarize(path):
    d=read(path);pairs=d['restriction_pairs'];groups=[]
    for mode in ('host','graph'):
        rows=[r for r in pairs if r['mode']==mode];ratios=[]
        for pair in (1,2,3):
            a=next(r for r in rows if r['pair']==pair and r['restriction']=='serial')
            b=next(r for r in rows if r['pair']==pair and r['restriction']=='warp')
            ratios.append(a['solve_ms']/b['solve_ms'])
        groups.append({'execution':mode,'paired_solve_speedups':ratios,'median_solve_speedup':statistics.median(ratios),
            'arms':{arm:{'median_solve_ms':statistics.median(r['solve_ms'] for r in rows if r['restriction']==arm),
                         'iterations':[r['iterations'] for r in rows if r['restriction']==arm],
                         'max_delta_to_first_serial':max(r['serial_difference']['relative'] for r in rows if r['restriction']==arm),
                         'true_residual_range':[min(r['true_relative_residual'] for r in rows if r['restriction']==arm),
                                                max(r['true_relative_residual'] for r in rows if r['restriction']==arm)]}
                    for arm in ('serial','warp')},
            'max_relative_solution_difference':max(r['serial_difference']['relative'] for r in rows)})
    return {'study':str(path),'sha256':sha(path),'frame':d['frame'],'outer':d.get('outer'),'inner':d.get('inner'),
            'linear_system_id':d.get('linear_system_id'),'operator_unchanged':d['system_unchanged'],
            'main_solution_restored_bitwise':d['primary_restored_bitwise'],'comparisons':groups}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--output',required=True);a=p.parse_args()
    rows=[summarize(p) for p in sorted((ROOT/a.directory).glob('*_study.json'))]
    result={'scope':'Default-rho PCG solve including Graph replay; common matrix/MAS preparation excluded; not whole-scene or certified speedup',
            'performance_certified':False,'systems':rows,
            'integrity_passed':bool(rows) and all(r['operator_unchanged'] and r['main_solution_restored_bitwise'] for r in rows)}
    with (ROOT/a.output).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
