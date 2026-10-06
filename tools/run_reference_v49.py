"""Bound one accurate solve on the actual unfinished-outer capture."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
prefix=ROOT/'runs/local/v49_bunny_late/events/e1';target=ROOT/'reports/REFERENCE_V49_LATE.json'
record=ROOT/'reports/REFERENCE_V49_LATE_PROCESS.json';assert not target.exists() and not record.exists()
command=[sys.executable,str(ROOT/'tools/reference_event_v47.py'),str(prefix),'--output',str(target)]
start=time.monotonic()
with (ROOT/'reports/REFERENCE_V49_LATE.log').open('x') as log:
    try:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        status={'status':'completed' if result.returncode==0 else 'failed','returncode':result.returncode}
    except subprocess.TimeoutExpired:status={'status':'timeout','returncode':None}
status.update(command=command,timeout_seconds=120,wall_seconds=time.monotonic()-start,
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in prefix.parent.glob(prefix.name+'_*')},
    worker_sha256=hashlib.sha256((ROOT/'tools/reference_event_v47.py').read_bytes()).hexdigest())
record.write_text(json.dumps(status,indent=2));print(json.dumps({k:v for k,v in status.items() if k!='input_sha256'}))
if target.exists():print(target.read_text())
