import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
rows=[]
for run in sorted((root/'runs/autodl').glob('autodl_perf_v39_*')):
    result=json.loads((run/'result.json').read_text())
    frames=json.loads((run/'output/stats.json').read_text())['frames']
    row={'run':run.name,**result,'stats_frames':len(frames),
         'accepted_per_frame':[sum('alpha' in t for t in f.get('toi',[])) for f in frames],
         'safe_states':len(list((run/'trace/substeps').glob('safe_*.bin'))),
         'frame_states':len(list((run/'trace').glob('state_*.bin')))}
    audits=[]
    for path in run.glob('mas_audit_*_audit.json'):
        j=json.loads(path.read_text())
        audits.append({'file':path.name,'frame':j.get('frame'),
                       'relative_repeat':max(max(q['repeat_relative']) for q in j['probes']),
                       'true_relative_residual':j.get('true_relative_residual'),
                       'keys':list(j)})
    row['audits']=audits;rows.append(row)
target=root/'reports/v39_run_summary.json';assert not target.exists()
target.write_text(json.dumps(rows,indent=2))
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k!='accepted_per_frame'}))
