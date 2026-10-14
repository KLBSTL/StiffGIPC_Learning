"""Reuse committed same-binary inputs after a launcher path fix, with provenance."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
import run_innovation_p2a as p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--prefix-parent',type=Path,required=True)
    parser.add_argument('--smoke-parent',type=Path,default=p.SESSION)
    a=parser.parse_args();session=a.session.resolve();prefix_parent=a.prefix_parent.resolve();smoke_parent=a.smoke_parent.resolve()
    prefixes=p.read(prefix_parent/'prefix_ledger.json');smoke=p.read(smoke_parent/'smoke_ledger.json')
    p.require(prefixes['complete'] and smoke['complete'],'No complete captured inputs')
    p.seal(session);packet=p.read(session/'seal.json')
    inherited=[]
    for source,ledger,stage in [(smoke_parent,smoke,'smoke'),(prefix_parent,prefixes,'prefix')]:
        parent=p.read(source/'seal.json')
        p.require(parent['identity']['exe']==packet['identity']['exe'],'Cross-binary input refused')
        p.require(parent['power_guid']==packet['power_guid'],'Power differs from inherited inputs')
        for row in ledger['runs']:
            folder=source/row['task']['name']
            p.require(row['passed'] and p.read(folder/'result.json')['status']=='completed','Inherited failed measurement')
            inherited.extend([p.record(folder/n) for n in ('requested.json','result.json','config_validation.json','p2_validation.json')])
        p.write_new(session/(stage+'_ledger.json'),ledger|dict(inherited_from=p.record(source/(stage+'_ledger.json'))))
    copies=[]
    for task in p.tasks('prefix'):
        source=prefix_parent/task['name']/'checkpoint';dest=session/task['name']/'checkpoint'
        meta=p.read(source/'checkpoint.json')
        p.require(meta['binary_id']==packet['identity']['exe']['sha256'] and meta['next_physical_frame']==task['save_checkpoint'],
            'Checkpoint frame/binary differs')
        before=[p.record(f) for f in sorted(source.iterdir()) if f.is_file()]
        dest.parent.mkdir(parents=True);shutil.copytree(source,dest)
        for original in before:
            copy=p.record(dest/Path(original['path']).name)
            p.require(copy['sha256']==original['sha256'] and copy['bytes']==original['bytes'],'Input copy differs')
            copies.append(dict(original=original,copy=copy))
    p.write_new(session/'resume_inputs.json',dict(controller=p.record(Path(__file__)),
        reason='Absolute native checkpoint-load path fix; no simulation or checkpoint identity change',
        inherited=inherited,copies=copies,checkpoint_metadata_rewritten=False))

if __name__=='__main__':main()
