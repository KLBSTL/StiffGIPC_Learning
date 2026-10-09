"""Pack unchanged real MAS inputs for the production-kernel equivalence test."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/local'))
sys.path.insert(0, str(ROOT/'tools/bench'))
from config import read
from linux_runner import require, write_new
from local_identity import record
from cpu_fixed_reference import fnv1a64


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',type=Path,required=True)
    args = p.parse_args()
    destination = args.capture/'real_mas_inputs.bin'
    require(not destination.exists(), 'Fresh packet required')
    sources = []
    data = []
    for metadata in sorted(args.capture.glob('*/fixed/*_meta.json')):
        meta = read(metadata)
        owners = [v for v in meta['local_preconditioners'] if v['kind']=='MAS_full_owned_buffers']
        require(len(owners)==1 and not owners[0]['cholesky'], 'One legacy MAS owner required')
        expected = owners[0]['buffers']['d_inverseMatMas']
        path = Path(str(metadata)[:-len('_meta.json')]+'_mas_d_inverseMatMas.bin')
        raw = path.read_bytes()
        require(expected['item_bytes']==9792 and len(raw)==expected['count']*9792
            and fnv1a64(raw)==expected['fnv1a64'], 'Captured MAS input fingerprint/ABI differs')
        sources.append(dict(metadata=record(metadata),input=record(path),matrices=expected['count']))
        data.append(raw)
    require(sources and sum(s['matrices'] for s in sources)<=16384, 'Packet outside native fixture budget')
    with destination.open('xb') as stream:
        for raw in data:
            stream.write(raw)
    write_new(args.capture/'real_mas_inputs.json',dict(sources=sources,packet=record(destination),
        matrices=sum(s['matrices'] for s in sources),packing='Concatenated identical bytes; no numerical transformation.'))
    print(dict(matrices=sum(s['matrices'] for s in sources),systems=len(sources),packet=str(destination)))


if __name__=='__main__':
    main()
