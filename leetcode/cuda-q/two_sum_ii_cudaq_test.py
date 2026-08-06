import sys
import unittest
from collections import Counter
from importlib import import_module
from pathlib import Path

import cudaq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cuda"))
cpu = import_module("two_sum_ii")
gpu = import_module("two_sum_ii_cuda")

from two_sum_ii_cudaq import (
    CANDIDATES,
    NUMBERS,
    TARGET,
    correct_candidate_index,
    two_sum_search,
)


class TwoSumCudaqTest(unittest.TestCase):
    def test_candidates_cover_every_pair(self):
        n = len(NUMBERS)
        self.assertEqual(len(CANDIDATES), n * (n - 1) // 2)
        self.assertEqual(set(CANDIDATES), {(i, j) for i in range(n) for j in range(i + 1, n)})

    def test_exactly_one_candidate_matches_target(self):
        matches = [(i, j) for i, j in CANDIDATES if NUMBERS[i] + NUMBERS[j] == TARGET]
        self.assertEqual(len(matches), 1)

    def test_correct_candidate_index_sums_to_target(self):
        i, j = CANDIDATES[correct_candidate_index()]
        self.assertEqual(NUMBERS[i] + NUMBERS[j], TARGET)

    def test_quantum_search_top_answer_matches_cpu_and_gpu(self):
        cudaq.set_target("nvidia")
        cudaq.set_random_seed(2)

        correct_index = correct_candidate_index()
        bit2, bit1, bit0 = bool(correct_index & 4), bool(correct_index & 2), bool(correct_index & 1)

        results = cudaq.run(two_sum_search, bit2, bit1, bit0, 2, shots_count=1000)
        counts = Counter(
            (4 if r[0] else 0) + (2 if r[1] else 0) + (1 if r[2] else 0)
            for r in results
        )
        top_index = counts.most_common(1)[0][0]
        i, j = CANDIDATES[top_index]
        quantum_answer = [i + 1, j + 1]

        self.assertEqual(quantum_answer, cpu.two_sum(NUMBERS, TARGET))
        self.assertEqual(quantum_answer, gpu.two_sum_gpu(NUMBERS, TARGET))


if __name__ == "__main__":
    unittest.main()
