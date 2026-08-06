import sys
import unittest
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
cpu = import_module("two_sum_ii")

from two_sum_ii_cuda import two_sum_gpu


class TwoSumGpuTest(unittest.TestCase):
    def test_example_1(self):
        self.assertEqual(two_sum_gpu([2, 7, 11, 15], 9), [1, 2])

    def test_example_2(self):
        self.assertEqual(two_sum_gpu([2, 3, 4], 6), [1, 3])

    def test_example_3(self):
        self.assertEqual(two_sum_gpu([-1, 0], -1), [1, 2])

    def test_answer_at_the_ends(self):
        self.assertEqual(two_sum_gpu([1, 2, 3, 4, 5], 9), [4, 5])

    def test_no_solution_raises(self):
        with self.assertRaises(ValueError):
            two_sum_gpu([1, 2, 3], 100)

    def test_matches_cpu_reference(self):
        cases = [
            ([2, 7, 11, 15], 9),
            ([2, 3, 4], 6),
            ([-1, 0], -1),
            ([-3, -1, 0, 2, 4, 7], 3),
            ([1, 1, 3, 5], 8),
            (list(range(1, 51)), 99),
        ]
        for numbers, target in cases:
            with self.subTest(numbers=numbers, target=target):
                self.assertEqual(two_sum_gpu(numbers, target), cpu.two_sum(numbers, target))


if __name__ == "__main__":
    unittest.main()
