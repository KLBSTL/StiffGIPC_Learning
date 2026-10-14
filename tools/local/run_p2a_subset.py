"""Explicit fresh subset/retry, preserving a resource-stopped parent session."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/local'))
import run_innovation_p2a as p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('seal','prefix','cost','audit','matched','whole'))
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--scenes',nargs='+',choices=list(p.SCENES),required=True)
    parser.add_argument('--parent',type=Path,default=p.SESSION)
    a=parser.parse_args();a.session=a.session.resolve();a.parent=a.parent.resolve()
    p.require(len(set(a.scenes))==len(a.scenes),'Duplicate scene')
    p.SCENES={s:p.SCENES[s] for s in a.scenes}
    if a.stage=='seal':
        parent=p.read(a.parent/'seal.json');smoke=p.read(a.parent/'smoke_ledger.json')
        p.require(smoke['complete'],'No valid parent smoke')
        p.seal(a.session);packet=p.read(a.session/'seal.json')
        p.require(packet['identity']['exe']==parent['identity']['exe'],'Different native binary')
        p.verify(parent['identity']['sources']+[parent['identity']['exe']]+parent['identity']['dlls'])
        p.require(packet['power_guid']==parent['power_guid'],'Cannot inherit smoke across power change')
        p.write_new(a.session/'extension.json',dict(driver=p.record(Path(__file__)),scenes=a.scenes,
            parent_seal=p.record(a.parent/'seal.json'),parent_smoke=p.record(a.parent/'smoke_ledger.json'),
            reason='Explicit finite resource refinement; preserve all failed and partial parent measurements'))
        p.write_new(a.session/'smoke_ledger.json',smoke|dict(inherited_from=p.record(a.parent/'smoke_ledger.json')))
    else:
        e=p.read(a.session/'extension.json');p.require(e['scenes']==a.scenes,'Subset changed')
        p.verify([e['driver'],e['parent_seal'],e['parent_smoke']]);p.run_stage(a.session,a.stage)

if __name__=='__main__':main()
