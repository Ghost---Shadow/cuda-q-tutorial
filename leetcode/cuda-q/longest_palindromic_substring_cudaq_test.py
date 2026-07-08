import sys
import unittest
from collections import Counter
from importlib import import_module
from pathlib import Path

import cudaq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cuda"))
cpu = import_module("longest_palindromic_substring")
gpu = import_module("longest_palindromic_substring_cuda")

from longest_palindromic_substring_cudaq import (
    CANDIDATES,
    S,
    WEIGHTS,
    is_feasible,
    quantum_palindrome_search,
)


class LongestPalindromeCudaqTest(unittest.TestCase):
    def test_candidates_are_all_palindromic_substrings(self):
        for i, j in CANDIDATES:
            substring = S[i:j + 1]
            self.assertEqual(substring, substring[::-1])

    def test_weights_match_candidate_lengths(self):
        for (i, j), weight in zip(CANDIDATES, WEIGHTS):
            self.assertEqual(weight, j - i + 1)

    def test_is_feasible_allows_at_most_one_candidate(self):
        self.assertTrue(is_feasible(()))
        self.assertTrue(is_feasible((0,)))
        self.assertFalse(is_feasible((0, 1)))

    def test_quantum_search_top_answer_matches_cpu_and_gpu(self):
        cudaq.set_target("nvidia")
        cudaq.set_random_seed(42)

        results = cudaq.run(quantum_palindrome_search, 1, shots_count=1000)
        histogram = Counter(tuple(b) for b in results)
        top_bits = histogram.most_common(1)[0][0]
        top_subset = [k for k, b in enumerate(top_bits) if b]

        self.assertEqual(len(top_subset), 1)
        i, j = CANDIDATES[top_subset[0]]
        top_answer = S[i:j + 1]

        self.assertEqual(top_answer, cpu.longest_palindrome(S))
        self.assertEqual(top_answer, gpu.longest_palindrome_gpu(S))


if __name__ == "__main__":
    unittest.main()
