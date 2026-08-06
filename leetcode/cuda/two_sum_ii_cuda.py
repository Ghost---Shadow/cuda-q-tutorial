"""LeetCode 167: Two Sum II - Input Array Is Sorted - CUDA SIMT

Same problem as ../cpu/two_sum_ii.py, but instead of moving the CPU's
two-pointer's-worth of sequential state (which pointer to move depends on
the running sum so far - a real data dependency), this checks every
candidate pair (i, j) simultaneously on the GPU: n^2 independent threads,
no shared state except one result write when a thread finds the pair. Not
the asymptotically optimal approach (O(n^2) work instead of O(n)), but a
genuinely, honestly parallel one - there's no sequential substructure left
to exploit in a brute-force pairwise scan, unlike the diagonal-at-a-time
DP fill in the palindrome CUDA example.
"""

import cupy as cp

from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
cpu = import_module("two_sum_ii")

KERNEL_SOURCE = (Path(__file__).resolve().parent / "two_sum_pairs.cu").read_text()
TWO_SUM_PAIRS_KERNEL = cp.RawKernel(KERNEL_SOURCE, 'two_sum_pairs')


def two_sum_gpu(numbers: list[int], target: int) -> list[int]:
    n = len(numbers)
    numbers_gpu = cp.array(numbers, dtype=cp.int64)
    result_gpu = cp.array([-1, -1], dtype=cp.int32)

    threads_per_dim = 16
    blocks_per_dim = (n + threads_per_dim - 1) // threads_per_dim
    TWO_SUM_PAIRS_KERNEL(
        (blocks_per_dim, blocks_per_dim), (threads_per_dim, threads_per_dim),
        (numbers_gpu, n, cp.int64(target), result_gpu)
    )

    i, j = result_gpu.get()
    if i < 0:
        raise ValueError("no two numbers sum to target")
    return [int(i) + 1, int(j) + 1]


if __name__ == "__main__":
    tests = [
        ([2, 7, 11, 15], 9),
        ([2, 3, 4], 6),
        ([-1, 0], -1),
    ]
    for numbers, target in tests:
        gpu_result = two_sum_gpu(numbers, target)
        cpu_result = cpu.two_sum(numbers, target)
        print(f"{numbers}, target={target} -> GPU: {gpu_result}, CPU: {cpu_result}, "
              f"match: {gpu_result == cpu_result}")
