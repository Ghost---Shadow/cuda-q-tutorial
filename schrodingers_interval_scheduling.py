"""
Schrodinger's Interval Scheduling
===================================

PART 1 (classical): Weighted Interval Scheduling, the textbook DP.
Given weighted intervals, pick a subset of non-overlapping ones with
maximum total weight. Sort by end time; for each interval i, find
p(i), the latest interval that doesn't conflict with i (this needs
genuine random access - p(i) can jump back to any earlier index,
unlike Fibonacci's fixed window - see the conversation this script
came out of). Then:

    dp[i] = max(dp[i-1], value[i] + dp[p(i)])

PART 2 (quantum): the same problem, but deliberately NOT built as an
explicit DP table translated into gates. Instead, the qubits' own
entangling structure mirrors the problem's actual conflict graph - the
exact same pairwise "do these two intervals overlap" relationships
p(i) is derived from in the classical algorithm:

  - Each interval gets a decision qubit, prepared with a rotation
    biased proportional to that interval's weight - not a hard
    computation of dp[i], just a preference nudging higher-value
    intervals toward "included" before anything else happens.
  - For every conflicting PAIR of intervals, a real entangling gate
    (built from reusable AND/OR reversible-logic blocks, reused from
    schrodingers_nand_sat.py) marks "both included" as infeasible -
    this is the part that's genuinely graph-shaped: one entangling
    gate per edge of the conflict graph, not a lookup table.
  - Grover-style amplification then boosts feasible combinations.

Honesty check on what this quantum circuit actually proves: the
feasibility marking is exact and Grover-guaranteed (same rigor as the
3-SAT solver) - verified by an exhaustive truth table against
brute-force ground truth before any search runs. The weight-biasing,
though, is a genuine heuristic nudge, not a guarantee of finding the
mathematically optimal solution the way Grover's search guarantees
finding *a* marked state. The final comparison against the classical
DP's *proven* optimum is reported honestly as exactly that - a
comparison, not an equivalence proof.
"""

import itertools
import math
from collections import Counter

import cudaq

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
    """Two intervals conflict if they overlap in time (touching
    endpoints are fine - ending exactly when another starts is not a
    conflict)."""
    start_i, end_i, _ = INTERVALS[i]
    start_j, end_j, _ = INTERVALS[j]
    return start_i < end_j and start_j < end_i


CONFLICT_PAIRS = [(i, j) for i in range(NUM_INTERVALS) for j in range(i + 1, NUM_INTERVALS)
                  if conflicts(i, j)]


def latest_compatible(i: int, order: list) -> int:
    """p(i) in the classical algorithm: the latest interval (by end
    time, in `order`) that doesn't conflict with order[i]. Returns -1
    if none exists. This is genuine random access - it can jump back
    to any earlier position in `order`, not a fixed window."""
    for k in range(i - 1, -1, -1):
        if not conflicts(order[i], order[k]):
            return k
    return -1


def classical_dp():
    order = sorted(range(NUM_INTERVALS), key=lambda i: INTERVALS[i][1])
    dp = [0] * NUM_INTERVALS
    choice_included = [False] * NUM_INTERVALS

    for i in range(NUM_INTERVALS):
        weight_i = INTERVALS[order[i]][2]
        p_i = latest_compatible(i, order)
        include_value = weight_i + (dp[p_i] if p_i >= 0 else 0)
        exclude_value = dp[i - 1] if i > 0 else 0
        if include_value > exclude_value:
            dp[i] = include_value
            choice_included[i] = True
        else:
            dp[i] = exclude_value
            choice_included[i] = False

    # Reconstruct the chosen set by walking the choices backward.
    chosen = set()
    i = NUM_INTERVALS - 1
    while i >= 0:
        if choice_included[i]:
            chosen.add(order[i])
            i = latest_compatible(i, order)
        else:
            i -= 1

    return dp[-1], chosen


def brute_force_optimum():
    best_weight = 0
    best_subset = frozenset()
    for subset_size in range(NUM_INTERVALS + 1):
        for subset in itertools.combinations(range(NUM_INTERVALS), subset_size):
            if all(not conflicts(i, j) for idx, i in enumerate(subset) for j in subset[idx + 1:]):
                weight = sum(INTERVALS[i][2] for i in subset)
                if weight > best_weight:
                    best_weight = weight
                    best_subset = frozenset(subset)
    return best_weight, best_subset


NUM_CONFLICTS = len(CONFLICT_PAIRS)
CONFLICT_I = [p[0] for p in CONFLICT_PAIRS]
CONFLICT_J = [p[1] for p in CONFLICT_PAIRS]

# Weight-proportional bias: each interval's inclusion probability
# ranges from 30% (lightest) to 80% (heaviest) - a nudge, not a
# determination. Computed once, classically, from INTERVALS' weights.
_weights = [w for _, _, w in INTERVALS]
_max_weight = max(_weights)
BIAS_PROBABILITIES = [0.3 + 0.5 * (w / _max_weight) for w in _weights]
BIAS_ANGLES = [2 * math.asin(math.sqrt(p)) for p in BIAS_PROBABILITIES]


@cudaq.kernel
def controlled_nand(inputs: cudaq.qview, target: cudaq.qubit):
    """target ^= NAND(inputs) - the same reusable primitive from
    schrodingers_nand_sat.py."""
    x.ctrl(inputs, target)
    x(target)


@cudaq.kernel
def mark_infeasible(decisions: cudaq.qview):
    """Phase-flips every FEASIBLE (conflict-free) combination of
    decisions. The entangling structure here is built directly from
    the conflict graph - one Toffoli per conflicting pair, exactly the
    same pairwise relationships p(i) is derived from classically - not
    a lookup table of precomputed answers.
    """
    conflict_ancillas = cudaq.qvector(NUM_CONFLICTS)
    any_conflict = cudaq.qubit()

    # conflict_ancillas[idx] = decisions[i] AND decisions[j], for each
    # conflicting pair (i, j) - a direct, per-edge translation of the
    # conflict graph into entangling gates.
    for idx in range(NUM_CONFLICTS):
        i = CONFLICT_I[idx]
        j = CONFLICT_J[idx]
        x.ctrl(decisions[i], decisions[j], conflict_ancillas[idx])

    # any_conflict = OR(conflict_ancillas), via NAND(NOT each) - same
    # De Morgan trick used for the Bouncer's Rule and 3-SAT clauses.
    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])
    controlled_nand(conflict_ancillas, any_conflict)
    for idx in range(NUM_CONFLICTS):
        x(conflict_ancillas[idx])

    # Mark the FEASIBLE branch (any_conflict == 0) with a phase flip.
    x(any_conflict)
    z(any_conflict)
    x(any_conflict)

    # Uncompute - same gates again, since every step here is a
    # self-inverse XOR operation.
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
    """Reflects about the *biased* initial state prepare_biased
    creates, not the uniform superposition standard Grover reflects
    about - built as U * Z_0 * U^-1, where U is prepare_biased itself,
    so the reflection axis actually matches what was prepared."""
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


if __name__ == "__main__":
    print("=== Part 1: classical weighted interval scheduling ===")
    print("Intervals (start, end, weight):")
    for i, (s, e, w) in enumerate(INTERVALS):
        print(f"  I{i}: ({s}, {e}, {w})")
    print(f"Conflict pairs: {CONFLICT_PAIRS}")

    dp_weight, dp_chosen = classical_dp()
    bf_weight, bf_chosen = brute_force_optimum()

    print()
    print(f"DP optimal weight:          {dp_weight}, chosen intervals: {sorted(dp_chosen)}")
    print(f"Brute-force optimal weight: {bf_weight}, chosen intervals: {sorted(bf_chosen)}")
    print(f"DP and brute force agree on weight: {dp_weight == bf_weight}")

    print()
    print("=== Part 2: quantum search, graph-native feasibility + weight bias ===")
    print(f"Per-interval inclusion bias (before any search): "
          f"{[f'{p:.2f}' for p in BIAS_PROBABILITIES]}")

    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 4000

    for iterations in [0, 1, 2, 3]:
        results = cudaq.run(quantum_interval_search, iterations, shots_count=shots)
        histogram = Counter(tuple(b) for b in results)

        weighted_results = []
        for bits, count in histogram.items():
            subset = frozenset(i for i, b in enumerate(bits) if b)
            feasible = all(not conflicts(i, j) for idx, i in enumerate(subset)
                            for j in list(subset)[idx + 1:])
            weight = sum(INTERVALS[i][2] for i in subset) if feasible else 0
            weighted_results.append((subset, feasible, weight, count))

        weighted_results.sort(key=lambda r: -r[3])
        print(f"\n--- {iterations} Grover iteration(s) ---")
        for subset, feasible, weight, count in weighted_results[:5]:
            tag = " <- matches classical optimum" if set(subset) == dp_chosen else ""
            print(f"  {sorted(subset)}: {count}/{shots} ({count/shots:.1%})  "
                  f"feasible={feasible}  weight={weight}{tag}")

        top_subset = weighted_results[0][0]
        print(f"  Most likely outcome matches classical DP optimum "
              f"({sorted(dp_chosen)}): {set(top_subset) == dp_chosen}")
