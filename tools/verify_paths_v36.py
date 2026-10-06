"""Check exported accepted-path coverage and record the independent validator identity."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('ccd',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert not a.output.exists()
frames=json.loads((a.run/'output/stats.json').read_text())['frames'];segments=0
for i,f in enumerate(frames):
    accepted=[t for t in f['toi'] if 'alpha' in t]
    states=sorted((a.run/'trace/substeps').glob(f'safe_{i:04d}_*.bin'))
    assert len(states)==len(accepted)+1
    assert states[0].read_bytes()==(a.run/f'trace/state_{i:04d}.bin').read_bytes()
    assert states[-1].read_bytes()==(a.run/f'trace/state_{i+1:04d}.bin').read_bytes()
    segments+=len(accepted)
bridges=max(0,len(frames)-1)
# Validator sorts every safe snapshot and checks all adjacent pairs, including
# the stationary endpoint/startpoint bridge between consecutive physical frames.
ccd=json.loads(a.ccd.read_text());assert ccd['passed'] and ccd['paths_checked']==segments+bridges
exe=ROOT/'builds/validator/Release/validate_path.exe'
result={'run':str(a.run.resolve()),'frames':len(frames),'accepted_segments':segments,
        'stationary_frame_bridges':bridges,'validator_segments':segments+bridges,'frame_bridges_bitwise':True,
        'passed':True,'validator_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),
        'command':[str(exe),str((a.run/'trace').resolve()),str(a.ccd.resolve()),'substeps','--stable-nh1'],
        'ccd':ccd}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({'frames':len(frames),'accepted_segments':segments,'stationary_bridges':bridges,'validator_segments':segments+bridges,'passed':True}))
