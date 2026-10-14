"""Meaningful evidence-parser contracts: missing values, duplicates, frame and N/P consistency."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[2]/'tools/bench/m0_offline.py'
SPEC = importlib.util.spec_from_file_location('m0_offline', FILE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class EvidenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch=FILE.parents[2]/'build/m0_test_scratch'
        cls.scratch.mkdir(parents=True,exist_ok=True)

    def test_numeric_nonfinite_boolean_and_missing(self):
        for value in (None, '', True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                m.number(value, 'test')
        for value in (-1, 1.5):
            with self.assertRaises(ValueError):
                m.number(value, 'count', True)

    def test_quantiles_do_not_synthesize_missing(self):
        self.assertIsNone(m.distribution([])['p50'])
        self.assertEqual(m.distribution([4, 2])['p50'], 3)
        self.assertEqual(m.distribution([7])['p99'], 7)

    def test_json_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            m.parse_native('{"alpha": NaN}')

    def fixture(self, folder, corrupt=None):
        frames, newton = [], []
        for f in range(1, 106):
            phases = {p:1.0 for p in m.PHASES}
            frame = dict(run='performance_bunny_original_r1', scene='Bunny',method='original',repeat='1',
                physical_frame=f,local_frame=f,dt=.04,N=1,P=5,newton_records=1,exit_reason='movement',
                **{p+'_ms':v for p,v in phases.items()})
            native = dict(pcg=dict(iterations=5,frame=f),phase_ms=phases,alpha=1.0,beta=1.0,
                active_pairs=0,energy_evaluations=1,energy_backtracks=0,intersection_backtracks=0,
                initial_energy_reused=True,kappa_before=1.0,kappa_after=1.0)
            row = {k:frame[k] for k in ('run','scene','method','repeat','physical_frame','local_frame')}
            row.update(record_index_0=0,P=5,alpha=1,beta=1,native_json=json.dumps(native),
                **{p+'_ms':v for p,v in phases.items()})
            frames.append(frame);newton.append(row)
        if corrupt:
            corrupt(frames,newton)
        for name, rows in [('frames.csv',frames),('newton.csv',newton)]:
            with (folder/name).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        return folder/'newton.csv',folder/'frames.csv'

    def test_complete_frames_and_missing_coverage(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            report=m.analyze(*self.fixture(Path(temp)), 'fixture')
        self.assertEqual(report['validated_frames'],105)
        summary=report['first100_summary'][0]
        self.assertEqual(summary['solves'],100)
        self.assertEqual(summary['missing_native_fields']['alpha_geometric'],100)
        self.assertEqual(summary['first_trial_accepted_fraction'],1)

    def test_wrong_N_P_or_duplicate_are_rejected(self):
        cases=[lambda f,n:f[0].update(P=6),lambda f,n:n.append(dict(n[0])),
               lambda f,n:n[0].update(record_index_0=1),lambda f,n:f.pop()]
        for corrupt in cases:
            with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
                with self.assertRaises(ValueError):
                    m.analyze(*self.fixture(Path(temp),corrupt), 'fixture')

    def test_terminal_empty_P_is_not_a_missing_solve(self):
        def terminal(frames,newton):
            row=dict(newton[0]);phases={p:1.0 if p=='assembly' else 0.0 for p in m.PHASES}
            row.update(record_index_0=1,P='',alpha='',beta='',
                native_json=json.dumps(dict(exit='movement',phase_ms=phases)),
                **{p+'_ms':v for p,v in phases.items()})
            newton.insert(1,row);frames[0].update(newton_records=2,assembly_ms=2.0)
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            report=m.analyze(*self.fixture(Path(temp),terminal),'fixture')
        summary=report['first100_summary'][0]
        self.assertEqual(summary['solves'],100)
        self.assertEqual(summary['terminal_records'],1)
        self.assertEqual(summary['missing_native_fields']['alpha'],1)


if __name__=='__main__':
    unittest.main()
