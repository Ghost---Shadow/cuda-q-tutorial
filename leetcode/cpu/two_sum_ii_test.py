import unittest

from two_sum_ii import two_sum


class TwoSumTest(unittest.TestCase):
    def test_example_1(self):
        self.assertEqual(two_sum([2, 7, 11, 15], 9), [1, 2])

    def test_example_2(self):
        self.assertEqual(two_sum([2, 3, 4], 6), [1, 3])

    def test_example_3(self):
        self.assertEqual(two_sum([-1, 0], -1), [1, 2])

    def test_negative_and_positive(self):
        self.assertEqual(two_sum([-3, -1, 0, 2, 4, 7], 3), [2, 5])

    def test_answer_at_the_ends(self):
        self.assertEqual(two_sum([1, 2, 3, 4, 5], 6), [1, 5])

    def test_duplicates_in_array_not_part_of_solution(self):
        self.assertEqual(two_sum([1, 1, 3, 5], 8), [3, 4])

    def test_no_solution_raises(self):
        with self.assertRaises(ValueError):
            two_sum([1, 2, 3], 100)


if __name__ == "__main__":
    unittest.main()
