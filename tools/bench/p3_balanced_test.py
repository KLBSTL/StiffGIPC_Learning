import copy
import unittest
from p3_balanced_analyze import batch_summary


class BalancedTiming(unittest.TestCase):
    def sample(self):
        rows=[]
        for i in range(32):
            mode='W' if ((i%4 in (1,2)) != bool((i//4)%2)) else 'F'
            rows.append(dict(index=i,block=i//4,position=i%4,mode=mode,counts_equal=True,
                complete_cpu_ms=2. if mode=='F' else 1.,gpu_interval_ms=1. if mode=='F' else .5))
        a=dict(samples=rows,boundary_typed_checks=4,all_timed_counts_equal=True,gpu_events_injected=False)
        b=copy.deepcopy(a);b['gpu_events_injected']=True
        return dict(schema='gipc.p3.ee.balanced.v2',cpu_primary=a,gpu_separate=b)
    def test_balanced_metrics(self):
        d=batch_summary(self.sample())
        self.assertEqual(d['cpu']['paired_geometric_speedup'],2)
        self.assertEqual(d['cpu']['first_mode_speedup'],dict(F=2.,W=2.))
        self.assertEqual(len(d['gpu']['paired_ratios']),16)
    def test_corrupt_or_incomplete_not_accepted(self):
        for key,value in [('index',9),('mode','X'),('complete_cpu_ms',float('nan')),('counts_equal',False)]:
            d=self.sample();d['cpu_primary']['samples'][0][key]=value
            with self.assertRaises(Exception):batch_summary(d)
        d=self.sample();d['cpu_primary']['samples'].pop()
        with self.assertRaises(Exception):batch_summary(d)
    def test_events_cannot_pollute_primary(self):
        d=self.sample();d['cpu_primary']['gpu_events_injected']=True
        with self.assertRaises(Exception):batch_summary(d)


if __name__=='__main__':unittest.main()
