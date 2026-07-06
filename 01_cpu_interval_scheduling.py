"""
Weighted Interval Scheduling: CPU, single-threaded
=====================================================

The baseline. Sort by end time, then for each interval find p(i) - the
latest interval that doesn't conflict with it (genuine random access,
not a fixed window - p(i) can jump back to any earlier index) - and
apply the DP recurrence:

    dp[i] = max(dp[i-1], value[i] + dp[p(i)])

Computing p(i) for each interval is embarrassingly parallel (every
interval's search is independent of every other's) - that's what
02_cuda_simt_interval_scheduling.py moves onto the GPU. The dp[]
recurrence itself is NOT embarrassingly parallel (dp[i] depends on
dp[i-1] and dp[p(i)], a genuine sequential dependency chain), so it
stays sequential in every version of this script.

This script is the reference every later version gets checked against.
"""

# (start, end, weight) for each interval.
INTERVALS = [
    (1, 4, 3),   # I0
    (3, 5, 5),   # I1
    (0, 6, 4),   # I2
    (5, 7, 2),   # I3
    (6, 9, 6),   # I4
]
NUM_INTERVALS = len(INTERVALS)


def conflicts(i: int, j: int) -> bool:
    start_i, end_i, _ = INTERVALS[i]
    start_j, end_j, _ = INTERVALS[j]
    return start_i < end_j and start_j < end_i


def compute_p(order: list) -> list:
    """p(i) = index (within `order`) of the latest interval that
    doesn't conflict with order[i], or -1 if none exists. Computed
    here as a plain sequential loop over all i, one at a time."""
    p = []
    for i in range(len(order)):
        found = -1
        for k in range(i - 1, -1, -1):
            if not conflicts(order[i], order[k]):
                found = k
                break
        p.append(found)
    return p


def solve(order: list, p: list):
    dp = [0] * len(order)
    included = [False] * len(order)

    for i in range(len(order)):
        weight_i = INTERVALS[order[i]][2]
        include_value = weight_i + (dp[p[i]] if p[i] >= 0 else 0)
        exclude_value = dp[i - 1] if i > 0 else 0
        if include_value > exclude_value:
            dp[i] = include_value
            included[i] = True
        else:
            dp[i] = exclude_value
            included[i] = False

    chosen = set()
    i = len(order) - 1
    while i >= 0:
        if included[i]:
            chosen.add(order[i])
            i = p[i]
        else:
            i -= 1

    return dp[-1], chosen


if __name__ == "__main__":
    print("=== CPU, single-threaded ===")
    print("Intervals (start, end, weight):")
    for i, (s, e, w) in enumerate(INTERVALS):
        print(f"  I{i}: ({s}, {e}, {w})")

    order = sorted(range(NUM_INTERVALS), key=lambda i: INTERVALS[i][1])
    p = compute_p(order)
    print(f"\nSort order (by end time): {order}")
    print(f"p(i) for each position in that order: {p}")

    optimal_weight, chosen = solve(order, p)
    print(f"\nOptimal weight: {optimal_weight}")
    print(f"Chosen intervals: {sorted(chosen)}")
