"""Reuse sealed GPU inputs; add a real rank-deficient history control."""
import json
from pathlib import Path
import numpy as np
from m3_numeric import ROOT,need
from prepare_cgstats_gpu import absolute_payloads,writer
from local_identity import verify,record
from linux_runner import write_new

RAW=ROOT/'runs/innovation_p5_qrfree_20261016'
OLD=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(read(RAW/'CPU_summary.json')['passed'],'CPU gate required')
    for name in ('inputs','fixture_positive_inputs'):
        original=read(OLD/name/'manifest.json');verify(original['records']);m=absolute_payloads(original,OLD/name)
        folder=RAW/name;writer(folder,m)
        m.update(qrfree_rank=8,parent_manifest=record(OLD/name/'manifest.json'))
        write_new(folder/'manifest.json',m)
    original=read(RAW/'fixture_positive_inputs/manifest.json');m=absolute_payloads(original,RAW/'fixture_positive_inputs')
    put=writer(RAW/'fixture_negative_inputs',m);n=m['n'];steps=m['source_iterations']
    # Every pre-update x and the final x are collinear: S has rank one, while
    # positive scalar statistics still claim rank eight. Coarse E must reject.
    direction=np.linspace(.2,1.2,n);X=direction[:,None]*np.arange(steps)[None,:]
    put('iterates',X);put('final_source_x',direction*steps)
    m.update(test_rejection_only=True,expected_failure='QR-free coarse rank/conditioning rejected',
             negative_control='rank-one history with full-rank statistics')
    write_new(RAW/'fixture_negative_inputs/manifest.json',m)
    print(json.dumps(dict(prepared=True,rank=8,inputs=3)))


if __name__=='__main__':main()
