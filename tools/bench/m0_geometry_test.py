"""Rest-state Jacobian convention: a deformed checkpoint is never the rest denominator."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('m0_analyze',ROOT/'tools/bench/m0_analyze.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class GeometryContracts(unittest.TestCase):
    def test_inverted_checkpoint_remains_inverted_against_pristine_rest(self):
        rest=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        current=rest.copy();current[3,2]=-1
        tets=np.array([[0,1,2,3]])
        self.assertEqual(m.jacobians(current,rest,tets)[0],-1)
        self.assertEqual(m.jacobians(current,current,tets)[0],1)

    def test_zero_rest_volume_rejected(self):
        rest=np.zeros((4,3));tets=np.array([[0,1,2,3]])
        with self.assertRaises(ValueError):m.jacobians(rest,rest,tets)

    def test_fixed_ABD_is_not_free_when_point_boundary_is_zero(self):
        scratch=ROOT/'build/m0_test_scratch';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            folder=Path(temp);(folder/'trace').mkdir();(folder/'output').mkdir()
            np.array([5,1,1,0,1,2,0,1,2,3],dtype='<u4').tofile(folder/'trace/topology.bin')
            np.zeros(5,dtype='<i4').tofile(folder/'trace/boundary_types.bin')
            (folder/'trace/metadata.json').write_text(json.dumps({'abd_point_num':4}))
            (folder/'output/scene.json').write_text(json.dumps({'objects':[{'body_type':'ABD','fixed_mode':'all'}]}))
            *_,masks=m.layout(folder)
            self.assertEqual(masks['free'].tolist(),[False,False,False,False,True])
            self.assertEqual(int(masks['fixed'].sum()),4)


if __name__=='__main__':unittest.main()
