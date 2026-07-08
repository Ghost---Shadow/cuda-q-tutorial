import unittest

from longest_palindromic_substring import longest_palindrome


class LongestPalindromeTest(unittest.TestCase):
    def test_odd_length_palindrome(self):
        result = longest_palindrome("babad")
        self.assertIn(result, ("bab", "aba"))

    def test_even_length_palindrome(self):
        self.assertEqual(longest_palindrome("cbbd"), "bb")

    def test_single_character(self):
        self.assertEqual(longest_palindrome("a"), "a")

    def test_no_repeating_characters(self):
        result = longest_palindrome("ac")
        self.assertIn(result, ("a", "c"))

    def test_empty_string(self):
        self.assertEqual(longest_palindrome(""), "")

    def test_whole_string_is_palindrome(self):
        self.assertEqual(longest_palindrome("racecar"), "racecar")

    def test_all_same_characters(self):
        self.assertEqual(longest_palindrome("aaaa"), "aaaa")

    def test_two_characters_same(self):
        self.assertEqual(longest_palindrome("aa"), "aa")

    def test_two_characters_different(self):
        result = longest_palindrome("ab")
        self.assertIn(result, ("a", "b"))

    def test_palindrome_at_start(self):
        self.assertEqual(longest_palindrome("abaxyz"), "aba")

    def test_palindrome_at_end(self):
        self.assertEqual(longest_palindrome("xyzaba"), "aba")

    def test_longer_mixed_string(self):
        result = longest_palindrome("forgeeksskeegfor")
        self.assertEqual(result, "geeksskeeg")


if __name__ == "__main__":
    unittest.main()
