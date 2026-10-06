"""CPU serialization/operator contracts; no native build or GPU invocation."""
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from cpu_fixed_reference import active_dofs, decimal_residual, expand_blocks, fnv1a64


class FormatTests(unittest.TestCase):
    def test_column_major_and_nonsymmetric_diagonal(self):
        block = np.arange(1., 10.).reshape(3, 3)
        raw = block.flatten(order="F").astype("<f8").tobytes()
        decoded = np.frombuffer(raw, dtype="<f8").reshape(1, 3, 3).transpose(0, 2, 1)
        actual = expand_blocks(np.array([0]), np.array([0]), decoded, 3)
        np.testing.assert_array_equal(actual.toarray(), block)

    def test_lower_block_not_dropped_and_transposed(self):
        block = np.arange(1., 10.).reshape(3, 3)
        actual = expand_blocks(np.array([1]), np.array([0]), block[None], 6).toarray()
        np.testing.assert_array_equal(actual[3:, :3], block)
        np.testing.assert_array_equal(actual[:3, 3:], block.T)
        self.assertEqual(np.count_nonzero(actual[:3, :3]), 0)

    def test_duplicate_terms_sum(self):
        blocks = np.array([np.eye(3), 2*np.eye(3)])
        actual = expand_blocks(np.array([0, 0]), np.array([1, 1]), blocks, 6).toarray()
        np.testing.assert_array_equal(actual[:3, 3:], 3*np.eye(3))

    def test_zero_row_requires_matching_zero_column_and_rhs(self):
        matrix = csr_matrix(np.diag([0., 2., 3.]))
        active, zero = active_dofs(matrix, np.zeros(3))
        np.testing.assert_array_equal(zero, [0])
        np.testing.assert_array_equal(active, [False, True, True])
        with self.assertRaises(ValueError):
            active_dofs(matrix, np.array([1., 0., 0.]))
        with self.assertRaises(ValueError):
            active_dofs(csr_matrix([[0., 0.], [1., 2.]]), np.zeros(2))

    def test_decimal_cancellation_preserves_small_contribution(self):
        matrix = csr_matrix([[1e16, 1., -1e16]])
        residual, metrics = decimal_residual(matrix, np.array([1.]), np.ones(3))
        self.assertEqual(metrics["relative_l2"], 0)
        np.testing.assert_array_equal(residual, [0])

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            expand_blocks(np.array([1]), np.array([0]), np.eye(3)[None], 3)
        with self.assertRaises(ValueError):
            expand_blocks(np.array([0]), np.array([0]), np.full((1, 3, 3), np.nan), 3)
        with self.assertRaises(ValueError):
            decimal_residual(csr_matrix(np.eye(1)), np.ones(1), np.array([np.inf]))

    def test_empty_matrix(self):
        matrix = expand_blocks(np.array([], dtype=int), np.array([], dtype=int), np.empty((0,3,3)), 0)
        active, zero = active_dofs(matrix, np.array([]))
        self.assertEqual(active.size, 0)
        self.assertEqual(zero.size, 0)

    def test_fnv(self):
        self.assertEqual(fnv1a64(b""), 14695981039346656037)
        self.assertEqual(fnv1a64(b"hello"), 0xa430d84680aabd0b)


if __name__ == "__main__":
    unittest.main()
