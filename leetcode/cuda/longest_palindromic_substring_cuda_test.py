import sys
import unittest
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
cpu = import_module("longest_palindromic_substring")

from longest_palindromic_substring_cuda import longest_palindrome_gpu


class LongestPalindromeGpuTest(unittest.TestCase):
    def test_odd_length_palindrome(self):
        result = longest_palindrome_gpu("babad")
        self.assertIn(result, ("bab", "aba"))

    def test_even_length_palindrome(self):
        self.assertEqual(longest_palindrome_gpu("cbbd"), "bb")

    def test_single_character(self):
        self.assertEqual(longest_palindrome_gpu("a"), "a")

    def test_no_repeating_characters(self):
        result = longest_palindrome_gpu("ac")
        self.assertIn(result, ("a", "c"))

    def test_empty_string(self):
        self.assertEqual(longest_palindrome_gpu(""), "")

    def test_whole_string_is_palindrome(self):
        self.assertEqual(longest_palindrome_gpu("racecar"), "racecar")

    def test_all_same_characters(self):
        self.assertEqual(longest_palindrome_gpu("aaaa"), "aaaa")

    def test_two_characters_same(self):
        self.assertEqual(longest_palindrome_gpu("aa"), "aa")

    def test_two_characters_different(self):
        result = longest_palindrome_gpu("ab")
        self.assertIn(result, ("a", "b"))

    def test_palindrome_at_start(self):
        self.assertEqual(longest_palindrome_gpu("abaxyz"), "aba")

    def test_palindrome_at_end(self):
        self.assertEqual(longest_palindrome_gpu("xyzaba"), "aba")

    def test_longer_mixed_string(self):
        result = longest_palindrome_gpu("forgeeksskeegfor")
        self.assertEqual(result, "geeksskeeg")

    def test_matches_cpu_reference(self):
        cases = [
            "babad", "cbbd", "a", "ac", "", "racecar", "aaaa", "aa", "ab",
            "abaxyz", "xyzaba", "forgeeksskeegfor",
            "the quick brown fox jumps over the lazy dog",
        ]
        for s in cases:
            with self.subTest(s=s):
                self.assertEqual(longest_palindrome_gpu(s), cpu.longest_palindrome(s))


if __name__ == "__main__":
    unittest.main()
