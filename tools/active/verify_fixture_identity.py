"""Carry CPU checks across a diagnostic-only rebuild only when GPU buffers match exactly."""
import argparse
import json
from config import ROOT,sha
from fixtures import CASES

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--output',required=True);a=p.parse_args()
    rows=[]
    for case,*_ in CASES:
        for file in ['factors.bin']+[f'p1_v{k}.bin' for k in range(10)]:
            old,new=[ROOT/folder/case/file for folder in [a.before,a.after]]
            hashes=[sha(old),sha(new)]
            rows.append({'case':case,'file':file,'before_sha256':hashes[0],'after_sha256':hashes[1],'bitwise_equal':hashes[0]==hashes[1]})
    passed=all(r['bitwise_equal'] for r in rows)
    with (ROOT/a.output).open('x') as f:json.dump({'passed':passed,'checks':rows},f,indent=2)
    print(json.dumps({'passed':passed,'buffers':len(rows)}));raise SystemExit(int(not passed))
