"""CPU-only synthetic ledger tests; no GPU, remote access, or package creation."""
import copy
import unittest

from autodl_prepare import make_plan
from autodl_window_chunks import (DRIVER_SCHEMA, check_requested_repeat, check_window_plan,
                                  smoke_failures, validate_prefix)


class ChunkLedger(unittest.TestCase):
    def setUp(self):
        candidate = {'preset': 'toi', 'mas_restrict': 'warp', 'mas_factor_action': 'factor_inverse'}
        self.plan = make_plan(candidate, 'window', triangular_reference=True)
        self.manifest = {'source_digest': 'source'}
        self.header = {'driver_schema': DRIVER_SCHEMA, 'driver_sha256': 'driver', 'plan_sha256': 'plan',
                       'candidate_sha256': self.plan['candidate_sha256'], 'source_digest': 'source',
                       'runs': [], 'next_plan_index': 0, 'completed_repeats': [], 'failed_arms': []}

    def validate(self, report, failed=()):
        return validate_prefix(report, self.plan, self.manifest, 'plan', 'driver', set(failed))

    def prefix(self, end, failed=(), fail_index=None):
        result = copy.deepcopy(self.header); blocked = set(failed)
        for index, task in enumerate(self.plan['runs'][:end]):
            if task['arm'] in blocked:
                continue
            status = 'timeout' if index == fail_index else 'completed'
            result['runs'].append(task | {'result': {'status': status}})
            if status != 'completed':
                blocked.add(task['arm'])
        result.update(next_plan_index=end, failed_arms=sorted(blocked),
                      completed_repeats=[r for r in (1, 2, 3) if end >= 18 * r])
        return result

    def test_original_plan_and_one_repeat(self):
        check_window_plan(self.plan)
        self.assertEqual(check_requested_repeat(0, 1), 18)
        self.assertEqual(check_requested_repeat(9, 1), 18)
        self.assertEqual(check_requested_repeat(18, 2), 36)
        self.assertEqual(check_requested_repeat(36, 3), 54)
        for cursor, repeat in [(0, 2), (18, 1), (19, 3), (54, 3)]:
            with self.assertRaises(ValueError):
                check_requested_repeat(cursor, repeat)

    def test_order_identity_and_result_preservation(self):
        report = self.prefix(18)
        self.assertEqual(self.validate(report), (18, set()))
        for key in ('driver_sha256', 'source_digest', 'candidate_sha256', 'plan_sha256'):
            changed = copy.deepcopy(report); changed[key] = 'other'
            with self.assertRaises(ValueError):
                self.validate(changed)
        swapped = copy.deepcopy(report); swapped['runs'][0], swapped['runs'][1] = swapped['runs'][1], swapped['runs'][0]
        with self.assertRaises(ValueError):
            self.validate(swapped)
        with self.assertRaises(ValueError):
            validate_prefix(report, self.plan, self.manifest, 'plan', 'driver', set(), lambda name: {'status': 'failed'})

    def test_failure_is_never_repeated(self):
        failed = {self.plan['runs'][1]['arm']}
        report = self.prefix(54, failed, fail_index=2)
        cursor, blocked = self.validate(report, failed)
        self.assertEqual(cursor, 54)
        self.assertEqual(len(report['runs']), 49)
        self.assertIn(self.plan['runs'][2]['arm'], blocked)
        matching = [r for r in report['runs'] if r['arm'] == self.plan['runs'][2]['arm']]
        self.assertEqual(len(matching), 1)
        report['runs'].append(self.plan['runs'][2] | {'result': {'status': 'completed'}})
        with self.assertRaises(ValueError):
            self.validate(report, failed)

    def test_smoke_requires_complete_matching_original(self):
        candidate = {'preset': 'toi', 'mas_restrict': 'warp', 'mas_factor_action': 'factor_inverse'}
        smoke_plan = make_plan(candidate, 'smoke', triangular_reference=True)
        smoke = {'plan_sha256': 'smoke', 'candidate_sha256': self.plan['candidate_sha256'],
                 'source_digest': 'source', 'runs': [task | {'result': {'status': 'completed'}} for task in smoke_plan['runs']]}
        smoke['runs'][0]['result']['status'] = 'failed'
        self.assertEqual(smoke_failures(smoke, smoke_plan, 'smoke', self.plan, self.manifest), {smoke['runs'][0]['arm']})
        smoke['runs'].pop()
        with self.assertRaises(ValueError):
            smoke_failures(smoke, smoke_plan, 'smoke', self.plan, self.manifest)


if __name__ == '__main__':
    unittest.main()
