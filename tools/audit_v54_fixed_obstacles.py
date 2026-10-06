"""Supplement old boundary-only checks with explicit fixed ABD object semantics."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for matrix in ['V54_CLOTH_MATRIX','V54_DT_MATRIX','V54_EXIT_MATRIX']:
    for r in json.loads((ROOT/f'reports/{matrix}.json').read_text()):
        if r['scene'] not in ['cloth_sphere7_l','cloth_fixed_bunny_l']:continue
        run=ROOT/'runs/local'/r['name'];scene=json.loads((run/'output/scene.json').read_text())
        assert all(o['body_type']=='ABD' and o['fixed_mode']=='all' for o in scene['objects'] if o['dimension']==3)
        raw=np.fromfile(run/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);tets=raw[3+nf*3:].reshape(-1,4);indices=np.unique(tets)
        x0=np.fromfile(run/'trace/state_0000.bin',dtype='<f8').reshape(-1,3);maximum=0.
        for i in range(1,r['result']['recorded_frames']+1):
            x=np.fromfile(run/f'trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
            maximum=max(maximum,float(np.linalg.norm(x[indices]-x0[indices],axis=1).max()))
        assert maximum<=1e-12
        rows.append({'run':r['name'],'fixed_abd_vertices':len(indices),'frames':r['result']['recorded_frames'],'max_motion_m':maximum})
out={'reason':'boundary_types marks FEM constraints but these fixed ABD vertices have zero boundary type; old fixed-mask check was empty for these scenes',
     'all_fixed_obstacles_checked':True,'scope':'All exported endpoints of v54 cloth, dt and exit matrices, two fixed-obstacle scenes','runs':rows}
target=ROOT/'reports/V54_FIXED_ABD_SUPPLEMENT.json';assert not target.exists();target.write_text(json.dumps(out,indent=2))
print(json.dumps({'runs':len(rows),'frames':sum(r['frames'] for r in rows),'max_motion_m':max(r['max_motion_m'] for r in rows)}))
