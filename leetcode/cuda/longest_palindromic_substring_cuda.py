"""LeetCode 5: Longest Palindromic Substring - CUDA SIMT

Same problem, same DP recurrence as
../cpu/longest_palindromic_substring.py:

    dp[i][j] = (s[i] == s[j]) and (j - i < 2 or dp[i + 1][j - 1])

but dp[][] is now filled by a real CUDA kernel. For a fixed substring
length, every dp[i][j] on that diagonal depends only on dp[i+1][j-1]
from the *previous* diagonal (length - 2) - never on another cell of
the same diagonal. That makes each diagonal embarrassingly parallel:
one GPU thread per starting index i, all running simultaneously.

The loop over length itself stays sequential on the host, on purpose
- it's a real data dependency (diagonal `length` needs diagonal
`length - 2` already written), the same reason the outer dp[]
recurrence stayed sequential in the interval-scheduling CUDA example.
This script only moves the genuinely parallel part (one diagonal's
worth of dp[][] entries) to the GPU.
"""

import cupy as cp

from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
cpu = import_module("longest_palindromic_substring")

KERNEL_SOURCE = (Path(__file__).resolve().parent / "palindrome_diagonal.cu").read_text()
PALINDROME_DIAGONAL_KERNEL = cp.RawKernel(KERNEL_SOURCE, 'palindrome_diagonal')


def longest_palindrome_gpu(s: str) -> str:
    n = len(s)
    if n == 0:
        return ""

    s_gpu = cp.array([ord(c) for c in s], dtype=cp.int8)
    dp_gpu = cp.zeros((n, n), dtype=cp.uint8)

    threads_per_block = 32
    for length in range(1, n + 1):
        count = n - length + 1
        blocks = (count + threads_per_block - 1) // threads_per_block
        PALINDROME_DIAGONAL_KERNEL((blocks,), (threads_per_block,),
                                    (s_gpu, dp_gpu, n, length))

    dp = dp_gpu.get()

    start, end = 0, 0
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if dp[i][j] and length > end - start + 1:
                start, end = i, j

    return s[start:end + 1]


if __name__ == "__main__":
    tests = ["babad", "cbbd", "a", "ac", "", "forgeeksskeegfor"]
    for t in tests:
        gpu_result = longest_palindrome_gpu(t)
        cpu_result = cpu.longest_palindrome(t)
        print(f"{t!r} -> GPU: {gpu_result!r}, CPU: {cpu_result!r}, "
              f"match: {gpu_result == cpu_result}")
