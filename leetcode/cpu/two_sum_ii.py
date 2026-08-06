"""LeetCode 167: Two Sum II - Input Array Is Sorted

Given a 1-indexed array sorted in non-decreasing order, find the two
indices whose values sum to target. Exactly one solution is guaranteed.

Two pointers, exploiting the sortedness: start at both ends. If the sum is
too small, the only way to grow it is to move the left pointer up
(everything to its right is >= it); if too big, move the right pointer
down. O(n) time, O(1) extra space - the constraint the problem asks for.
"""


def two_sum(numbers: list[int], target: int) -> list[int]:
    left, right = 0, len(numbers) - 1
    while left < right:
        total = numbers[left] + numbers[right]
        if total == target:
            return [left + 1, right + 1]
        if total < target:
            left += 1
        else:
            right -= 1
    raise ValueError("no two numbers sum to target")


if __name__ == "__main__":
    tests = [
        ([2, 7, 11, 15], 9),
        ([2, 3, 4], 6),
        ([-1, 0], -1),
    ]
    for numbers, target in tests:
        print(f"{numbers}, target={target} -> {two_sum(numbers, target)}")
