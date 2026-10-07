from pathlib import Path
import sqlite3
import tempfile
import unittest
import report_profile_attribution as a

class ProfileAttributionTests(unittest.TestCase):
    def fixture(self,folder,ambiguous=False):
        path=Path(folder)/'nsight.sqlite';c=sqlite3.connect(path)
        c.executescript('''
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,correlationId INTEGER,globalTid INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,correlationId INTEGER,graphNodeId INTEGER,demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER);
            INSERT INTO StringIds VALUES(1,'cub::DeviceReduceSingleTile<double>()');
            INSERT INTO NVTX_EVENTS VALUES(0,1000000,'ipc.physical_frame',NULL,5);
            INSERT INTO NVTX_EVENTS VALUES(10,20,'ipc.energy_batch_reduce',NULL,5);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(11,12,1,5);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100,200,1,0,1);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(300,400,1,9,1);
            INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(300,500);
        ''')
        if ambiguous:c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(13,14,1,5)')
        c.commit();c.close()

    def test_unique_runtime_owner_and_union_without_double_counting(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder);d=a.analyze(Path(folder))
            self.assertEqual(d['energy_kernel_owners']['ipc.energy_batch_reduce']['count'],1)
            self.assertAlmostEqual(d['gpu_work_union_including_whole_graph']['union_ms'],.0003)
            self.assertAlmostEqual(d['whole_graph']['union_ms'],.0002)
            self.assertFalse(d['graph_internal_operator_attribution_available'])

    def test_ambiguous_correlation_not_attributed(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder,True);d=a.analyze(Path(folder))
            self.assertEqual(d['energy_kernel_owners'],{})
            self.assertEqual(d['outside_graph_without_unique_runtime_count'],1)

if __name__=='__main__':unittest.main()
