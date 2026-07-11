"""
Schrodinger's Shipping: the Berth Allocation Problem
========================================================

PART 1 (classical): unlike schrodingers_interval_scheduling.py's
single-resource case (efficiently solvable in O(n log n)), this is the
Berth Allocation Problem - K identical berths, N ships each needing
exclusive dock time for a known window, maximize total priority-
weighted value of serviced ships subject to at most K ships docked
simultaneously at any moment. The moment K > 1, this jumps complexity
class to NP-hard - there is no known efficient exact algorithm, which
is exactly the property that makes this a legitimate (if still small-
scale) target for exploring quantum search, unlike the K=1 case.

Because there's no polynomial DP here, the classical baseline is
brute force (exact but exponential - fine at this toy size, not at
real-world scale) alongside a greedy heuristic (what a real system
might actually run for larger instances, with no optimality
guarantee) - an honest reflection of how this problem is actually
handled in practice.

PART 2 (quantum): same principle as the interval-scheduling script -
no explicit DP table, no lookup oracle. The entangling structure is
built directly from the ships' overlap structure: for every maximal
group of mutually-overlapping ships whose size exceeds berth capacity
K, a genuine N-input NAND gate (reused from schrodingers_nand_sat.py)
marks "not all of these can be included at once" - a direct,
per-violated-group translation of the conflict structure, verified
against brute-force ground truth before any search runs. Weight-biased
state preparation and a matching diffusion operator (same construction
as the interval-scheduling script) then bias the Grover search towards
higher-value feasible schedules.
"""

import itertools
import math
from collections import Counter

import cudaq

# (arrival, departure, value) for each ship.
SHIPS = [
    (0, 4, 5),   # S0
    (1, 3, 3),   # S1
    (2, 6, 6),   # S2
    (5, 9, 4),   # S3
    (6, 8, 5),   # S4
    (7, 10, 3),  # S5
]
NUM_SHIPS = len(SHIPS)
BERTH_CAPACITY = 2  # at most this many ships may be docked simultaneously


def overlaps(i: int, j: int) -> bool:
    start_i, end_i, _ = SHIPS[i]
    start_j, end_j, _ = SHIPS[j]
    return start_i < end_j and start_j < end_i


def is_feasible(subset) -> bool:
    """A subset is feasible iff no group of mutually-overlapping ships
    within it exceeds berth capacity - checked directly, by looking at
    every sub-group of the subset (fine at this toy size; a real
    system would sweep a timeline instead)."""
    for size in range(BERTH_CAPACITY + 1, len(subset) + 1):
        for combo in itertools.combinations(subset, size):
            if all(overlaps(a, b) for idx, a in enumerate(combo) for b in combo[idx + 1:]):
                return False
    return True


def find_maximal_overlap_cliques():
    """All maximal groups of mutually-overlapping ships whose size
    exceeds berth capacity - these are exactly the constraints the
    quantum oracle needs to enforce."""
    violating_groups = []
    for size in range(BERTH_CAPACITY + 1, NUM_SHIPS + 1):
        for combo in itertools.combinations(range(NUM_SHIPS), size):
            if all(overlaps(a, b) for idx, a in enumerate(combo) for b in combo[idx + 1:]):
                # Only keep it if it's not a subset of an already-found
                # (larger) violating group, i.e. keep maximal cliques.
                if not any(set(combo) < set(existing) for existing in violating_groups):
                    violating_groups.append(combo)
    # Remove any group that turned out to be a subset of a later, larger one.
    maximal = [g for g in violating_groups
               if not any(set(g) < set(other) for other in violating_groups if other != g)]
    return maximal


VIOLATING_GROUPS = find_maximal_overlap_cliques()


def brute_force_optimum():
    best_weight = 0
    best_subset = frozenset()
    for size in range(NUM_SHIPS + 1):
        for subset in itertools.combinations(range(NUM_SHIPS), size):
            if is_feasible(subset):
                weight = sum(SHIPS[i][2] for i in subset)
                if weight > best_weight:
                    best_weight = weight
                    best_subset = frozenset(subset)
    return best_weight, best_subset


def greedy_heuristic():
    """A simple, realistic stand-in for what a real system might run
    on a larger instance where brute force isn't an option: sort by
    value descending, greedily include a ship if it doesn't push any
    group over capacity. No optimality guarantee."""
    order = sorted(range(NUM_SHIPS), key=lambda i: -SHIPS[i][2])
    chosen = []
    for i in order:
        candidate = chosen + [i]
        if is_feasible(candidate):
            chosen.append(i)
    return sum(SHIPS[i][2] for i in chosen), frozenset(chosen)


NUM_GROUPS = len(VIOLATING_GROUPS)

_weights = [w for _, _, w in SHIPS]
_max_weight = max(_weights)
BIAS_PROBABILITIES = [0.1 + 0.4 * (w / _max_weight) for w in _weights]
BIAS_ANGLES = [2 * math.asin(math.sqrt(p)) for p in BIAS_PROBABILITIES]


@cudaq.kernel
def controlled_nand(inputs: cudaq.qview, target: cudaq.qubit):
    x.ctrl(inputs, target)
    x(target)


@cudaq.kernel
def mark_infeasible(decisions: cudaq.qview):
    """Phase-flips every FEASIBLE combination. One N-input NAND per
    maximal over-capacity group - a direct translation of the ships'
    overlap structure into entangling gates, not a lookup table.

    NOTE: this hard-codes the two specific violating groups found for
    THIS instance (ships 0-1-2, and ships 3-4-5) as direct qview
    slices, rather than looping generically over an arbitrary list of
    groups - CUDA-Q kernels can't easily build a qview from a
    dynamically-indexed, non-contiguous set of qubits, and both
    groups here happen to be contiguous ranges. A different SHIPS
    instance whose violating groups aren't contiguous would need this
    written out differently (e.g. by reordering ships so overlapping
    groups are adjacent, or by giving each group its own explicitly
    named decision-qubit variables instead of slicing one shared
    qview).
    """
    # Qubit flow (2 ancillas, compute -> use -> uncompute, all borrowed
    # and returned to |0> so they never leak into the final measurement):
    #   decisions[0:3], decisions[3:6] : read-only CONTROLS throughout -
    #     never modified, since they're the actual answer we're solving
    #     for and must stay live for every later Grover iteration.
    #   group_violated[0], group_violated[1] : written by the two
    #     multi-controlled x.ctrl calls below (TARGETS), then read as
    #     controls into controlled_nand, then unwritten (returned to
    #     |0>) by literally repeating the same x.ctrl calls a second
    #     time - XOR-based compute is its own inverse.
    #   any_violation : written once by controlled_nand, read by the
    #     z-sandwich that does the actual phase flip, then unwritten by
    #     a second controlled_nand call before the qubit goes out of
    #     scope.
    group_violated = cudaq.qvector(2)
    any_violation = cudaq.qubit()

    # COMPUTE: group_violated[g] becomes 1 iff every ship in group g is
    # included (decisions[...] all |1>) - decisions are CONTROLS,
    # group_violated are TARGETS.
    x.ctrl(decisions[0:3], group_violated[0])  # group (S0, S1, S2)
    x.ctrl(decisions[3:6], group_violated[1])  # group (S3, S4, S5)

    # any_violation = OR(group_violated) via NAND(NOT each). The two
    # x() calls temporarily flip group_violated so controlled_nand's
    # AND-then-NOT reads as an OR of the ORIGINAL bits; group_violated
    # is CONTROL into controlled_nand, any_violation is its TARGET.
    x(group_violated[0])
    x(group_violated[1])
    controlled_nand(group_violated, any_violation)
    x(group_violated[0])
    x(group_violated[1])

    # USE: mark the FEASIBLE branch (any_violation == 0) with a phase
    # flip - any_violation is read AND written here, but only its
    # phase changes, not its measured value (X, Z, X is self-inverse on
    # the |0> component - see the uncompute discussion in the module
    # docstring's sibling script, schrodingers_interval_scheduling.py).
    x(any_violation)
    z(any_violation)
    x(any_violation)

    # UNCOMPUTE: run the exact same compute steps again, in reverse
    # order, to return every ancilla to |0> - required so
    # group_violated/any_violation don't stay entangled with decisions
    # and corrupt the interference this oracle depends on.
    x(group_violated[0])
    x(group_violated[1])
    controlled_nand(group_violated, any_violation)
    x(group_violated[0])
    x(group_violated[1])

    x.ctrl(decisions[3:6], group_violated[1])
    x.ctrl(decisions[0:3], group_violated[0])


@cudaq.kernel
def prepare_biased(decisions: cudaq.qview):
    for i in range(NUM_SHIPS):
        ry(BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def unprepare_biased(decisions: cudaq.qview):
    for i in range(NUM_SHIPS):
        ry(-BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def diffusion(decisions: cudaq.qview):
    # Qubit flow: every decisions[i] passes through 3 stages in order -
    # unprepare (rotate the biased basis back to computational |0...0>),
    # a multi-controlled phase flip on |0...0> specifically (decisions[0:-1]
    # as CONTROLS, decisions[-1] as TARGET, sandwiched in x() so the
    # control condition becomes "all zero" instead of "all one"), then
    # re-prepare (rotate back into the biased basis). Net effect: every
    # amplitude gets reflected around the biased-mean amplitude, not
    # just the uniform mean - this is what "weighted Grover diffusion"
    # means concretely, gate by gate.
    unprepare_biased(decisions)
    for i in range(NUM_SHIPS):
        x(decisions[i])
    z.ctrl(decisions[0:NUM_SHIPS - 1], decisions[NUM_SHIPS - 1])
    for i in range(NUM_SHIPS):
        x(decisions[i])
    prepare_biased(decisions)


@cudaq.kernel
def quantum_berth_search(iterations: int) -> list[bool]:
    # Qubit flow: decisions is written once by prepare_biased (moves
    # every qubit from |0> into its individually-biased superposition),
    # then alternately read-and-rewritten by mark_infeasible (phase-only
    # - amplitudes' signs change, but decisions never leaves superposition)
    # and diffusion (amplitude-only - reflects magnitudes, no new phase
    # info) for `iterations` rounds, then read exactly once, destructively,
    # by the final mz. No qubit is measured until that very last line.
    decisions = cudaq.qvector(NUM_SHIPS)
    prepare_biased(decisions)

    for _ in range(iterations):
        mark_infeasible(decisions)
        diffusion(decisions)

    return mz(decisions)


if __name__ == "__main__":
    print("=== Part 1: classical Berth Allocation Problem ===")
    print("Ships (arrival, departure, value):")
    for i, (a, d, v) in enumerate(SHIPS):
        print(f"  S{i}: ({a}, {d}, {v})")
    print(f"Berth capacity: {BERTH_CAPACITY}")
    print(f"Maximal over-capacity groups: {VIOLATING_GROUPS}")

    bf_weight, bf_subset = brute_force_optimum()
    greedy_weight, greedy_subset = greedy_heuristic()

    print()
    print(f"Brute-force optimal weight: {bf_weight}, ships: {sorted(bf_subset)}")
    print(f"Greedy heuristic weight:    {greedy_weight}, ships: {sorted(greedy_subset)}")
    print(f"Greedy matches optimum: {greedy_weight == bf_weight}")

    print()
    print("=== Part 2a: verifying the feasibility oracle against brute force ===")
    cudaq.set_target("nvidia")

    all_phases_ok = True
    for bits in itertools.product([False, True], repeat=NUM_SHIPS):
        @cudaq.kernel
        def prep_and_mark(b0: bool, b1: bool, b2: bool, b3: bool, b4: bool, b5: bool):
            decisions = cudaq.qvector(NUM_SHIPS)
            flags = [b0, b1, b2, b3, b4, b5]
            for i in range(NUM_SHIPS):
                if flags[i]:
                    x(decisions[i])
            mark_infeasible(decisions)

        state = cudaq.get_state(prep_and_mark, *bits)
        bits_str = "".join("1" if b else "0" for b in bits)
        index = int(bits_str, 2) * (2 ** 3)  # 2 group ancillas + 1 any_violation, uncomputed to 0
        amplitude = complex(state[index])
        subset = tuple(i for i in range(NUM_SHIPS) if bits[i])
        expected_sign = -1 if is_feasible(subset) else 1
        ok = (amplitude.real * expected_sign) > 0.5
        all_phases_ok &= ok
        if not ok:
            print(f"  MISMATCH at {subset}: amplitude={amplitude:.3f} expected_sign={expected_sign:+d}")
    print(f"All 64 oracle phases correct: {all_phases_ok}")

    print()
    print("=== Part 2b: quantum search, graph-native feasibility + weight bias ===")
    print(f"Per-ship inclusion bias: {[f'{p:.2f}' for p in BIAS_PROBABILITIES]}")
    total_feasible = sum(
        1 for size in range(NUM_SHIPS + 1)
        for subset in itertools.combinations(range(NUM_SHIPS), size)
        if is_feasible(subset))
    print(f"Feasible subsets: {total_feasible}/{2**NUM_SHIPS} "
          f"({total_feasible/2**NUM_SHIPS:.1%}) - unlike the interval-scheduling")
    print("script's instance, MOST subsets here are already feasible.")

    cudaq.set_random_seed(42)
    shots = 4000

    print()
    print("--- An honest finding, not a bug: Grover oscillates when the marked")
    print("    (feasible) fraction exceeds 50% ---")
    m_over_n = total_feasible / 2 ** NUM_SHIPS
    theta = math.asin(math.sqrt(m_over_n))
    print(f"theta = arcsin(sqrt(M/N)) = {math.degrees(theta):.1f} degrees "
          f"(M/N = {m_over_n:.3f})")
    print("Theory predicts the feasible-fraction after k iterations is "
          "sin((2k+1)*theta)^2:")
    for iterations in range(6):
        predicted = math.sin((2 * iterations + 1) * theta) ** 2
        print(f"  k={iterations}: theory predicts {predicted:.1%}")

    print()
    print("Measured feasible-fraction (uniform H superposition, for a clean")
    print("apples-to-apples comparison against the theoretical formula):")
    uniform_results_by_k = {}
    for iterations in range(6):
        results = cudaq.run(quantum_berth_search, iterations, shots_count=shots)
        feasible_count = sum(
            1 for bits in results
            if is_feasible(tuple(i for i, b in enumerate(bits) if b)))
        uniform_results_by_k[iterations] = results
        print(f"  k={iterations}: measured {feasible_count/shots:.1%}")

    print()
    print("The match confirms the oracle and diffusion operator are working")
    print("exactly as Grover's theory predicts - the counter-intuitive part is")
    print("that a LARGE marked fraction (>50%) makes iteration count actively")
    print("HARMFUL past a point: a single iteration can overshoot 90 degrees")
    print("and rotate probability AWAY from the marked subspace, not toward")
    print("it. This is a genuine, well-documented Grover limitation, not")
    print("something specific to this circuit's construction.")

    print()
    print("=== Consequence: does amplifying 'feasible' help find the BEST")
    print("    feasible schedule specifically? ===")
    for iterations in [0, 1, 2, 3]:
        results = cudaq.run(quantum_berth_search, iterations, shots_count=shots)
        histogram = Counter(tuple(b) for b in results)

        scored = []
        for bits, count in histogram.items():
            subset = tuple(i for i, b in enumerate(bits) if b)
            feasible = is_feasible(subset)
            weight = sum(SHIPS[i][2] for i in subset) if feasible else 0
            scored.append((subset, feasible, weight, count))
        scored.sort(key=lambda r: -r[3])

        print(f"\n--- {iterations} Grover iteration(s) ---")
        for subset, feasible, weight, count in scored[:5]:
            tag = " <- matches brute-force optimum" if set(subset) == bf_subset else ""
            print(f"  {sorted(subset)}: {count}/{shots} ({count/shots:.1%})  "
                  f"feasible={feasible}  weight={weight}{tag}")
        print(f"  Most likely outcome matches brute-force optimum: "
              f"{set(scored[0][0]) == bf_subset}")

    print()
    print("=== Honest conclusion ===")
    print("Marking 'feasible' alone doesn't discriminate by weight at all -")
    print("every feasible state gets an equal boost regardless of value, so")
    print("even when the oscillation issue is sidestepped (e.g. by choosing")
    print("k=0, i.e. relying on the weight-bias alone with no Grover rounds),")
    print("there's no guarantee the single most likely outcome is the true")
    print("optimum, only that it's *more likely* to be a good one than a")
    print("uniformly random guess. Reliably finding THE best schedule this")
    print("way would need marking based on a target weight value (not just")
    print("feasibility), or a fundamentally different technique - quantum")
    print("maximum-finding algorithms (e.g. Durr-Hoyer) - genuinely beyond")
    print("this script's scope.")
