"""Check the declared material policy on an analytic interior-inversion path."""
import json,subprocess,tempfile,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
folder=Path(tempfile.mkdtemp(prefix='material-policy-',dir=ROOT/'builds/validator'))
trace=folder/'trace';trace.mkdir()
x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype='<f8')
y=x@np.diag([-2.,-.5,1.])
np.array([4,0,1,0,1,2,3],dtype='<u4').tofile(trace/'topology.bin')
np.zeros(4,dtype='<i4').tofile(trace/'body_ids.bin')
np.zeros(4,dtype='<i4').tofile(trace/'boundary_types.bin')
(trace/'metadata.json').write_text(json.dumps({'abd_point_num':0}))
x.tofile(trace/'state_0000.bin');y.tofile(trace/'state_0001.bin')
exe=ROOT/'builds/validator/Release/validate_path.exe'
strict=folder/'strict.json';stable=folder/'stable.json'
a=subprocess.run([str(exe),str(trace),str(strict)],capture_output=True,text=True)
b=subprocess.run([str(exe),str(trace),str(stable),'--stable-nh1'],capture_output=True,text=True)
sa=json.loads(strict.read_text());sb=json.loads(stable.read_text())
assert a.returncode==1 and not sa['passed'] and sa['tet_inversions']==1
assert b.returncode==0 and sb['passed'] and sb['tet_inversions']==1
assert sa['conservative_collision_flags']==sb['conservative_collision_flags']==0
assert sa['positive_volume_required'] and not sb['positive_volume_required']
rigid_trace=folder/'rigid_trace';shutil.copytree(trace,rigid_trace)
(rigid_trace/'metadata.json').write_text(json.dumps({'abd_point_num':4}))
rigid=folder/'rigid.json'
c=subprocess.run([str(exe),str(rigid_trace),str(rigid),'--stable-nh1'],capture_output=True,text=True)
sc=json.loads(rigid.read_text())
assert c.returncode==1 and sc['abd_tet_inversions']==1 and not sc['passed']
print(json.dumps({'passed':True,'scope':'declared FEM material gate; ABD orientation and external CCD remain required','fixture':str(folder)}))
