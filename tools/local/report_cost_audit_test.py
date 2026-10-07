import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import report_cost_audit as c
import report_rounds
from config import digest

class CostAuditTests(unittest.TestCase):
    def test_only_diagnostic_fields_change(self):
        config=report_rounds.task('sphere','combined',1,'screen')['config']
        values={'requested.json':dict(expanded_config=config,config_sha256=digest(config)),
                'result.json':dict(status='completed',recorded_frames=100,finite=True)}
        with patch.object(c,'read',side_effect=lambda p:values[p.name]):
            _,observed=c.derive(Path('fixture'))
        self.assertEqual({k for k in config if config[k]!=observed[k]},
                         {'diagnostics','cost_frames','cost_events'})
        self.assertEqual(observed['cost_frames'],'1-100')

    def test_nested_envelopes_not_added_to_physical_time(self):
        rows=[]
        for frame in range(1,101):
            root=2*frame-1
            for stage,scope,parent,ms in (('ipc.physical_frame',root,0,10.),
                                         ('ipc.energy_batch_reduce',root+1,root,1.)):
                rows.append(dict(stage=stage,scope_id=scope,parent_scope_id=parent,frame=frame,
                    cpu_submit_ms=ms,measurement_mode='nvtx_cpu_only',gpu_events_enabled=False,
                    sample_kind='production'))
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'cost.jsonl';p.write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
            result=c.summarize(p)
            self.assertEqual(result['stages']['ipc.physical_frame']['cpu_inclusive_ms'],1000.)
            self.assertEqual(result['stages']['ipc.energy_batch_reduce']['fraction_of_physical_cpu'],.1)
            rows.pop();rows.pop()
            p.write_text('\n'.join(map(json.dumps,rows)),encoding='utf-8')
            with self.assertRaises(ValueError):c.summarize(p)

if __name__=='__main__':unittest.main()
