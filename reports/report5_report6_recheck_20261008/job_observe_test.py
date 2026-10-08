"""CPU-only fault tests of the production owned-Job observation boundary."""
import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/local"))
from windows_owned_job import OwnedJob


class FakeApi:
    def __init__(self, *, image=True, alive=False, inside=True, open_handle=True, wait_error=False):
        self.image, self.alive, self.inside = image, alive, inside
        self.open_handle, self.wait_error = open_handle, wait_error
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        return 123 if self.open_handle else 0

    def IsProcessInJob(self, handle, job, output):
        C.cast(output, C.POINTER(W.BOOL))[0] = self.inside
        return 1

    def GetProcessTimes(self, handle, created, exited, kernel, user):
        C.cast(created, C.POINTER(W.FILETIME))[0].dwLowDateTime = 345
        return 1

    def QueryFullProcessImageNameW(self, handle, flags, output, size):
        if self.image:
            output.value = "test-owned.exe"
            return 1
        C.set_last_error(5)
        return 0

    def WaitForSingleObject(self, handle, timeout):
        if self.wait_error:
            C.set_last_error(6)
            return 0xffffffff
        return 258 if self.alive else 0

    def GetExitCodeProcess(self, handle, output):
        C.cast(output, C.POINTER(W.DWORD))[0] = 4
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


@unittest.skipUnless(os.name == "nt", "ctypes Windows error API required; no GPU needed")
class ObservationBoundary(unittest.TestCase):
    def make_job(self, **options):
        job = object.__new__(OwnedJob)
        job.api = FakeApi(**options)
        job.job, job.members = 777, {}
        job.pids = lambda: {789}
        return job

    def test_live_image_failure_is_not_relaxed(self):
        job = self.make_job(image=False, alive=True)
        with self.assertRaises(PermissionError):
            job.observe()
        self.assertEqual(job.members, {})
        self.assertEqual(job.api.closed, [123])

    def test_exited_member_retains_kernel_identity_and_error(self):
        job = self.make_job(image=False)
        row, = job.observe()
        self.assertEqual((row["pid"], row["creation_time_100ns"], row["exit_code"]), (789, 345, 4))
        self.assertIsNone(row["image"])
        self.assertEqual(row["image_query_error"], 5)
        self.assertTrue(row["exited_before_image_query"])
        self.assertEqual(job.api.closed, [])
        # No duplicate identity or stale replacement on a subsequent sample.
        self.assertEqual(job.observe(), [row])
        self.assertEqual(job.api.closed, [123])

    def test_unknown_wait_state_fails_closed(self):
        job = self.make_job(image=False, wait_error=True)
        with self.assertRaises(OSError):
            job.observe()
        self.assertEqual(job.members, {})
        self.assertEqual(job.api.closed, [123])

    def test_successful_image_is_retained(self):
        job = self.make_job(image=True, alive=True)
        row, = job.observe()
        self.assertEqual(row["image"], "test-owned.exe")
        self.assertIsNone(row["exit_code"])
        self.assertEqual(row["image_query_error"], 0)
        self.assertFalse(row["exited_before_image_query"])
        self.assertEqual(job.api.closed, [])

    def test_pid_reused_outside_job_is_not_retained(self):
        job = self.make_job(inside=False)
        self.assertEqual(job.observe(), [])
        self.assertEqual(job.api.closed, [123])

    def test_member_already_gone_before_open(self):
        job = self.make_job(open_handle=False)
        self.assertEqual(job.observe(), [])
        self.assertEqual(job.api.closed, [])


if __name__ == "__main__":
    unittest.main()
