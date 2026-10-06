"""Append this matrix's evidence/decisions, preserving all prior index records."""
import json
from datetime import datetime,timezone
import index
from config import ROOT,read

if __name__=='__main__':
    path=ROOT/'reports/active/EXPERIMENT_INDEX.json';previous=read(path)['entries']
    smoke=read(ROOT/'reports/active/FOURWAY_CLOTH_SMOKE_BATCH_20261005.json')['runs']
    timing=read(ROOT/'reports/active/FOURWAY_CLOTH_TIMING_BATCH_20261005.json')['runs']
    targets={t['name']:t for t in smoke+timing};assert len(targets)==32
    assert not {r['id'] for r in previous}&targets.keys(),'Already indexed; do not overwrite curated evidence'
    data=read(ROOT/'reports/active/FOURWAY_CLOTH_RESULTS_20261005.json')
    index.main();catalog=read(path)
    assert catalog['entries'][:len(previous)]==previous
    for row in catalog['entries'][len(previous):]:
        if row['id'] not in targets:continue
        t=targets[row['id']]
        row['quality_certified']=False;row['performance_certified']=False
        row['evidence']+=['reports/active/FOURWAY_CLOTH_RESULTS_20261005.md','reports/active/FOURWAY_CLOTH_RESULTS_20261005.json']
        if t['repeat']=='smoke':
            row.update(status='accepted',status_scope='Startup/configuration only',reason='Requested four-way startup checks completed; not quality certification')
        elif t['variant']=='base':
            row.update(status='accepted',status_scope='Current frozen-Stiff diagnostic reference only',reason='100-frame timing and quality-repeat reference completed')
        elif t['variant']=='graph':
            row.update(status='pending',status_scope='Quality equivalence unresolved; measured timing retained',reason='Timing/configuration completed; strict repeat envelope exceeded, with repeat sensitivity documented')
        else:
            row.update(status='rejected',status_scope='Physical-quality acceptance only; requested timing completed',reason='TOI cloth stretch and trajectories exceed Stiff repeat envelope; defaults retained per user request')
    catalog['entries'].append({'id':'fourway_cloth_matrix_20261005','origin':'active_aggregate','status':'accepted',
        'status_scope':'Requested diagnostic matrix completed only; no physical-quality/performance certification',
        'frames_per_run':100,'timing_runs':24,'startup_runs':8,'paired_repeats':3,
        'performance_certified':False,'quality_certified':False,
        'paired_median_speedups':{k:{a:v['median_paired_speedup_over_base'] for a,v in s['summary'].items()} for k,s in data['scenes'].items()},
        'evidence':['reports/active/FOURWAY_CLOTH_RESULTS_20261005.md','reports/active/FOURWAY_CLOTH_RESULTS_20261005.json',
            'reports/active/FOURWAY_CLOTH_TIMING_BATCH_20261005.json','reports/active/FOURWAY_CLOTH_VISUAL_20261005.json']})
    catalog['updated_utc']=datetime.now(timezone.utc).isoformat()
    ids={r['id'] for r in catalog['entries']};assert len(ids)==len(catalog['entries'])
    for r in catalog['entries']:
        for p in r['evidence']:assert (ROOT/p).exists(),p
    path.write_text(json.dumps(catalog,indent=2))
    assert read(path)['entries'][:len(previous)]==previous
    print(json.dumps({'previous_preserved':len(previous),'current_entries':len(catalog['entries']),'evidence_paths_passed':True}))
