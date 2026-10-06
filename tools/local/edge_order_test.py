"""CPU contract failures for the bounded probe controller and configuration."""
import json
import tempfile
import unittest
from pathlib import Path
import edge_order_rounds as runner
from config import expand, environment, matches_requested
from windows_runner import final_status


class Contracts(unittest.TestCase):
    def test_final_elapsed_cannot_escape_deadline(self):
        self.assertEqual(final_status('running',0,120),'completed')
        for elapsed in (120.0001,float('inf'),float('nan'),0,-1):
            self.assertEqual(final_status('running',0,elapsed),'timeout')
        self.assertEqual(final_status('running',1,1),'failed')
        self.assertEqual(final_status('memory_budget',0,121),'memory_budget')
    def test_default_and_explicit_selection(self):
        self.assertEqual(expand({})['edge_query_order'], 'raw')
        for scene in ('fixed', 'hang'):
            t = runner.task(scene)
            self.assertEqual(t['config']['edge_query_order'], 'raw')
            self.assertFalse(t['config']['contact_pool'])
            self.assertEqual(t['config']['diagnostics'], ['edge_order'])
            e = environment(t['config'], Path('run'))
            self.assertEqual(e['GIPC_EDGE_QUERY_ORDER'], 'raw')
            self.assertEqual(e['GIPC_EDGE_ORDER_PROBE_FILE'], str(Path('run/edge_order_probe.jsonl')))

    def test_reject_bad_modes_and_probe_budget(self):
        cases = [{'edge_query_order': 'morton'}, {'backend': 'toi_al', 'edge_query_order': 'leaf'},
                 {'edge_order_probe_frames': '1'}, {'diagnostics': ['edge_order']},
                 {'diagnostics': ['edge_order'], 'edge_order_probe_frames': '1,1'},
                 {'diagnostics': ['edge_order'], 'edge_order_probe_frames': '1,'},
                 {'diagnostics': ['edge_order'], 'edge_order_probe_frames': '2'},
                 {'diagnostics': ['edge_order'], 'edge_order_probe_frames': '57', 'steps': 56}]
        for c in cases:
            with self.subTest(c=c), self.assertRaises(ValueError): expand(c)

    def test_old_default_config_stays_readable(self):
        c = expand({})
        del c['edge_query_order']; del c['edge_order_probe_frames']
        self.assertTrue(matches_requested(c, {}))

    def test_no_retry_unknown_or_missing_predecessor(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)
            runner.previous(p, 'fixed', 'seal', {}, None)
            with self.assertRaises(ValueError): runner.previous(p, 'hang', 'seal', {}, 'anything')
            (p / 'unknown').mkdir()
            with self.assertRaises(ValueError): runner.previous(p, 'fixed', 'seal', {}, None)
            with self.assertRaises(ValueError): runner.previous(p, 'hang', 'seal', {}, 'anything')

    def test_missing_mismatched_or_invalid_probe_evidence_rejected(self):
        t=runner.task('fixed')
        def row(frame):
            return {'frame':frame,'passed':True,'production_order':'raw','first_query_only':True,
                    'inputs_read_only':True,'production_scratch_untouched':True,'leaf_permutation_valid':True,
                    'per_face_mismatches':0,'invalid_or_unwritten_faces':0,'compared_faces':257,
                    'face_count':257,'edge_count':3,'raw_any':True,'leaf_any':True,'raw_diagnostic_any':True,
                    'timed_pairs':7,'warmups_per_mode':2,'speed_evidence':True,'per_face_hit_count':129,
                    'host_readback_ms':1,'diagnostic_host_ms':2,
                    'pairs':[{'index':i,'first':'raw' if i%2==0 else 'leaf',
                              'raw_kernel_ms':2,'leaf_kernel_ms':1} for i in range(7)]}
        with tempfile.TemporaryDirectory() as name:
            p=Path(name);file=p/'edge_order_probe.jsonl'
            def save(rows):file.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
            save([row(1),row(57)])
            self.assertEqual(runner.probe_analysis(p,t)[1]['median_speedup'],2)
            save([row(1)])
            with self.assertRaises(ValueError):runner.probe_analysis(p,t)
            for field,bad in [('per_face_mismatches',1),('invalid_or_unwritten_faces',1),
                              ('compared_faces',256),('leaf_any',False),('timed_pairs',8)]:
                r=row(57);r[field]=bad;save([row(1),r])
                with self.subTest(field=field),self.assertRaises(ValueError):runner.probe_analysis(p,t)
            for bad in (0,float('nan'),-1):
                r=row(57);r['pairs'][0]['leaf_kernel_ms']=bad;save([row(1),r])
                with self.assertRaises(ValueError):runner.probe_analysis(p,t)
            r=row(57);r['pairs'][1]['first']='raw';save([row(1),r])
            with self.assertRaises(ValueError):runner.probe_analysis(p,t)


if __name__ == '__main__': unittest.main()
