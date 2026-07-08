"""
Weighted Interval Scheduling: CUDA SIMT
==========================================

Same problem, same instance, same algorithm as
01_cpu_interval_scheduling.py - but p(i) is now computed by a real
CUDA kernel, one GPU thread per interval, all running simultaneously
(Single Instruction Multiple Threads). This is the part of the
algorithm that's genuinely embarrassingly parallel: interval i's
search for its latest compatible predecessor doesn't depend on any
other interval's search.

The dp[] recurrence itself is NOT parallelized here, on purpose - it
has a real sequential dependency (dp[i] needs dp[i-1] and dp[p(i)]
already computed), the same reason it stayed sequential in the CPU
version. This script only moves the genuinely parallel part to the
GPU, and reports p(i) and the final answer against
01_cpu_interval_scheduling.py's output as ground truth.
"""

import cupy as cp

from importlib import import_module

cpu = import_module("01_cpu_interval_scheduling")

INTERVALS = cpu.INTERVALS
NUM_INTERVALS = cpu.NUM_INTERVALS

COMPUTE_P_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_p(const int* starts, const int* ends, const int* order,
               int* p, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    int interval_i = order[i];
    int start_i = starts[interval_i];
    int end_i = ends[interval_i];

    int found = -1;
    for (int k = i - 1; k >= 0; k--) {
        int interval_k = order[k];
        int start_k = starts[interval_k];
        int end_k = ends[interval_k];
        // conflicts(i, k) == start_i < end_k && start_k < end_i
        bool conflict = (start_i < end_k) && (start_k < end_i);
        if (!conflict) {
            found = k;
            break;
        }
    }
    p[i] = found;
}
''', 'compute_p')


def compute_p_on_gpu(order: list) -> list:
    starts = cp.array([INTERVALS[i][0] for i in range(NUM_INTERVALS)], dtype=cp.int32)
    ends = cp.array([INTERVALS[i][1] for i in range(NUM_INTERVALS)], dtype=cp.int32)
    order_gpu = cp.array(order, dtype=cp.int32)
    p_gpu = cp.full(len(order), -1, dtype=cp.int32)

    threads_per_block = 32
    blocks = (len(order) + threads_per_block - 1) // threads_per_block
    COMPUTE_P_KERNEL((blocks,), (threads_per_block,),
                      (starts, ends, order_gpu, p_gpu, len(order)))

    return p_gpu.get().tolist()


if __name__ == "__main__":
    print("=== CUDA SIMT: p(i) computed in parallel, one GPU thread per interval ===")
    order = sorted(range(NUM_INTERVALS), key=lambda i: INTERVALS[i][1])
    print(f"Sort order (by end time): {order}")

    p_gpu = compute_p_on_gpu(order)
    p_cpu = cpu.compute_p(order)
    print(f"p(i) from GPU (parallel): {p_gpu}")
    print(f"p(i) from CPU (sequential, reference): {p_cpu}")
    print(f"GPU and CPU p(i) arrays match: {p_gpu == p_cpu}")

    # The dp[] recurrence stays sequential - genuine data dependency.
    optimal_weight, chosen = cpu.solve(order, p_gpu)
    ref_weight, ref_chosen = cpu.solve(order, p_cpu)

    print(f"\nOptimal weight (using GPU-computed p): {optimal_weight}, "
          f"chosen: {sorted(chosen)}")
    print(f"Optimal weight (using CPU-computed p, reference): {ref_weight}, "
          f"chosen: {sorted(ref_chosen)}")
    print(f"Matches CPU reference: {optimal_weight == ref_weight and chosen == ref_chosen}")
