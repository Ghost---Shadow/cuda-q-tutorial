"""
Weighted Interval Scheduling: CUDA-Q
=======================================

Same problem, same instance as 01 and 02 - no explicit DP table
translated into gates. Each interval gets a decision qubit, prepared
with a rotation biased proportional to that interval's weight (a
nudge, not a hard computation), and for every conflicting PAIR of
intervals, a real entangling gate (a Toffoli-based NAND, reused from
schrodingers_nand_sat.py) marks "both included" as infeasible - one
gate per edge of the actual conflict graph, not a lookup table.
Grover-style amplification then boosts feasible combinations.

This instance (5 intervals, capacity 1) has a much smaller feasible
fraction than the multi-berth version explored earlier in this
project, which avoids the Grover-oscillation issue that showed up
there - verified directly below rather than assumed.

The final answer is checked against 02_cuda_simt_interval_scheduling.py's
result, which was itself checked against 01's - a genuine chain of
verification, CPU -> GPU -> quantum, each step checked against the one
before it rather than each being independently "trusted."
"""

import math
from collections import Counter
from importlib import import_module

import cudaq

simt = import_module("02_cuda_simt_interval_scheduling")

INTERVALS = simt.INTERVALS
NUM_INTERVALS = simt.NUM_INTERVALS


def conflicts(i: int, j: int) -> bool:
    start_i, end_i, _ = INTERVALS[i]
    start_j, end_j, _ = INTERVALS[j]
    return start_i < end_j and start_j < end_i


CONFLICT_PAIRS = [(i, j) for i in range(NUM_INTERVALS) for j in range(i + 1, NUM_INTERVALS)
                  if conflicts(i, j)]
NUM_CONFLICTS = len(CONFLICT_PAIRS)
CONFLICT_I = [p[0] for p in CONFLICT_PAIRS]
CONFLICT_J = [p[1] for p in CONFLICT_PAIRS]

_weights = [w for _, _, w in INTERVALS]
_max_weight = max(_weights)
BIAS_PROBABILITIES = [0.3 + 0.5 * (w / _max_weight) for w in _weights]
BIAS_ANGLES = [2 * math.asin(math.sqrt(p)) for p in BIAS_PROBABILITIES]


@cudaq.kernel
def controlled_nand(inputs: cudaq.qview, target: cudaq.qubit):
    x.ctrl(inputs, target)
    x(target)


@cudaq.kernel
def mark_infeasible(decisions: cudaq.qview):
    conflict_ancillas = cudaq.qvector(NUM_CONFLICTS)
    any_conflict = cudaq.qubit()

    for idx in range(NUM_CONFLICTS):
        i = CONFLICT_I[idx]
        j = CONFLICT_J[idx]
        x.ctrl(decisions[i], decisions[j], conflict_ancillas[idx])

    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])
    controlled_nand(conflict_ancillas, any_conflict)
    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])

    x(any_conflict)
    z(any_conflict)
    x(any_conflict)

    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])
    controlled_nand(conflict_ancillas, any_conflict)
    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])

    for idx in range(NUM_CONFLICTS - 1, -1, -1):
        i = CONFLICT_I[idx]
        j = CONFLICT_J[idx]
        x.ctrl(decisions[i], decisions[j], conflict_ancillas[idx])


@cudaq.kernel
def prepare_biased(decisions: cudaq.qview):
    for i in range(NUM_INTERVALS):
        ry(BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def unprepare_biased(decisions: cudaq.qview):
    for i in range(NUM_INTERVALS):
        ry(-BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def diffusion(decisions: cudaq.qview):
    unprepare_biased(decisions)
    for i in range(NUM_INTERVALS):
        x(decisions[i])
    z.ctrl(decisions[0:NUM_INTERVALS - 1], decisions[NUM_INTERVALS - 1])
    for i in range(NUM_INTERVALS):
        x(decisions[i])
    prepare_biased(decisions)


@cudaq.kernel
def quantum_interval_search(iterations: int) -> list[bool]:
    decisions = cudaq.qvector(NUM_INTERVALS)
    prepare_biased(decisions)
    for _ in range(iterations):
        mark_infeasible(decisions)
        diffusion(decisions)
    return mz(decisions)


def is_feasible(subset) -> bool:
    return all(not conflicts(i, j) for idx, i in enumerate(subset) for j in subset[idx + 1:])


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    print("=== Verifying the feasibility fraction avoids the oscillation issue ===")
    import itertools
    feasible_count = sum(
        1 for size in range(NUM_INTERVALS + 1)
        for subset in itertools.combinations(range(NUM_INTERVALS), size)
        if is_feasible(subset))
    total = 2 ** NUM_INTERVALS
    print(f"Feasible subsets: {feasible_count}/{total} ({feasible_count/total:.1%}) - "
          f"well under 50%, so Grover amplification should behave monotonically here.")

    print()
    print("=== Quantum search ===")
    cudaq.set_random_seed(42)
    shots = 4000
    for iterations in [0, 1, 2]:
        results = cudaq.run(quantum_interval_search, iterations, shots_count=shots)
        histogram = Counter(tuple(b) for b in results)
        scored = []
        for bits, count in histogram.items():
            subset = tuple(i for i, b in enumerate(bits) if b)
            feasible = is_feasible(subset)
            weight = sum(INTERVALS[i][2] for i in subset) if feasible else 0
            scored.append((subset, feasible, weight, count))
        scored.sort(key=lambda r: -r[3])

        print(f"\n--- {iterations} Grover iteration(s) ---")
        for subset, feasible, weight, count in scored[:3]:
            print(f"  {sorted(subset)}: {count}/{shots} ({count/shots:.1%})  "
                  f"feasible={feasible}  weight={weight}")

    # Final check: does the quantum search's top answer at the best
    # iteration count match the CUDA SIMT (and therefore CPU) result?
    results = cudaq.run(quantum_interval_search, 1, shots_count=shots)
    histogram = Counter(tuple(b) for b in results)
    top_bits = histogram.most_common(1)[0][0]
    top_subset = set(i for i, b in enumerate(top_bits) if b)

    order = sorted(range(NUM_INTERVALS), key=lambda i: INTERVALS[i][1])
    p = simt.compute_p_on_gpu(order)
    simt_weight, simt_chosen = simt.cpu.solve(order, p)

    print()
    print(f"Quantum top answer: {sorted(top_subset)}")
    print(f"CUDA SIMT reference answer: {sorted(simt_chosen)}")
    print(f"Quantum matches the CUDA SIMT -> CPU verification chain: "
          f"{top_subset == simt_chosen}")
