"""Separate flushed frame evidence from the unfinished timeout frame."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];data=ROOT/'downloads/autodl_perf_v43_20261003';rows=[]
for name in ['chol_bunny','chol_bunny_host']:
    run=data/'runs/autodl'/f'autodl_perf_v43_{name}';result=json.loads((run/'result.json').read_text());frames=json.loads((run/'output/stats.json').read_text())['frames']
    log=(run/'run.log').read_text(errors='replace');growth=[{'capacity_before':int(a),'detected_pairs':int(b)} for a,b in re.findall(r'Growing CCD pair storage from (\d+) for (\d+) detected pairs',log)]
    last=frames[-1];outer=last.get('toi',[]);accepted=[t for t in outer if 'alpha' in t]
    paths=list((run/'trace/substeps').glob('safe_*.bin'))
    row={'name':name,'result':result,'flushed_frames':len(frames),'unfinished_frame_statistics_available':len(frames)>result['recorded_frames'],
         'ccd_pair_storage_growth':growth,'max_detected_pairs_in_log':max((g['detected_pairs'] for g in growth),default=0),
         'last_flushed_frame':{'toi_exit':last.get('toi_exit'),'accepted_outer':len(accepted),
             'minimum_alpha':min((t['alpha'] for t in accepted),default=None),
             'max_safe_ccd_broad_pairs':max((t.get('safe_ccd_broad_pairs',0) for t in outer),default=0),
             'max_active_update_broad_pairs':max((t.get('active_update_broad_pairs',0) for t in outer),default=0)},
         'exported_safe_states':len(paths),'terminal_failure_system_available':(run/'failure_system_meta.json').exists(),
         'log_pcg_breakdown': 'PCG numerical breakdown' in log,'log_cholesky_pivot_failure':'MAS Cholesky invalid pivot' in log,
         'caution':'SIGTERM timeout does not flush final-frame solver statistics or export a terminal A/b snapshot; absence in flushed stats is not a proof about every unfinished solve.'}
    rows.append(row);print(json.dumps(row),flush=True)
target=ROOT/'reports/TIMEOUT_V43_20261003.json';assert not target.exists();target.write_text(json.dumps(rows,indent=2))
