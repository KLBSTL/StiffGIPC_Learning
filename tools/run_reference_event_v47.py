"""Bound a single sparse-direct event reference and retain timeout evidence."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
prefix=ROOT/'runs/local/v47_bunny_graph/events/e4'
output=ROOT/'reports/REFERENCE_V47_E4.json'
record_path=ROOT/'reports/REFERENCE_V47_E4_PROCESS.json'
assert not output.exists() and not record_path.exists()
command=[sys.executable,str(ROOT/'tools/reference_event_v47.py'),str(prefix),'--output',str(output)]
start=time.monotonic()
with (ROOT/'reports/REFERENCE_V47_E4.log').open('x') as log:
    try:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        record={'status':'completed' if result.returncode==0 else 'failed','returncode':result.returncode}
    except subprocess.TimeoutExpired:
        record={'status':'timeout','returncode':None}
record.update(command=command,timeout_seconds=120,wall_seconds=time.monotonic()-start,
    input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in prefix.parent.glob(prefix.name+'_*')})
record_path.write_text(json.dumps(record,indent=2))
print(json.dumps({k:v for k,v in record.items() if k!='input_sha256'}))
if output.exists():print(output.read_text())
