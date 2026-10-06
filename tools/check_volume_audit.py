"""Analytic fixtures: endpoint inspection must not miss an interior inversion."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
fixtures=[('translation',np.eye(3),1.0),
          ('interior_zero',np.diag([-1.,-1.,1.]),0.),
          ('interior_negative',np.diag([-2.,-.5,1.]),-.125)]
base=Path(tempfile.mkdtemp(prefix='volume-audit-',dir=ROOT/'builds/validator'))
for name,transform,expected in fixtures:
    run=base/name;trace=run/'trace';steps=trace/'substeps';steps.mkdir(parents=True)
    initial=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype='<f8')
    end=initial@transform.T
    if name=='translation':end+=np.array([.2,-.3,.4])
    np.array([4,0,1,0,1,2,3],dtype='<u4').tofile(trace/'topology.bin')
    initial.tofile(trace/'state_0000.bin');initial.tofile(steps/'safe_0000_0000.bin');end.tofile(steps/'safe_0000_0001.bin')
    report=run/'report.json'
    subprocess.run([sys.executable,str(ROOT/'tools/volume_path_metrics.py'),str(run),'--last-frame','0','--output',str(report)],
                   check=True,capture_output=True,text=True)
    d=json.loads(report.read_text())
    assert abs(d['relative_jacobian_path_min']-expected)<1e-12,(name,d)
    assert d['volume_gate_for_selected_interval']==(expected>0),(name,d)
print(json.dumps({'analytic_fixtures':len(fixtures),'passed':True,'fixtures':str(base)}))
