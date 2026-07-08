"""LeetCode 5: Longest Palindromic Substring

Given a string s, return the longest palindromic substring in s.

Dynamic programming: dp[i][j] is True when s[i..j] is a palindrome.
dp[i][j] = (s[i] == s[j]) and (j - i < 2 or dp[i + 1][j - 1])
"""


def longest_palindrome(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""

    dp = [[False] * n for _ in range(n)]
    start, end = 0, 0

    for i in range(n):
        dp[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and (length == 2 or dp[i + 1][j - 1]):
                dp[i][j] = True
                if length > end - start + 1:
                    start, end = i, j

    return s[start:end + 1]


if __name__ == "__main__":
    tests = ["babad", "cbbd", "a", "ac", ""]
    for t in tests:
        print(f"{t!r} -> {longest_palindrome(t)!r}")
