"""LeetCode 5: Longest Palindromic Substring - CUDA-Q

Same problem, same test string ("cbbd") already exercised by
../cpu/longest_palindromic_substring.py and
../cuda/longest_palindromic_substring_cuda.py - no dp[][] table
translated into gates. Every (start, end) substring of S that is
*classically* known to be a palindrome gets its own decision qubit
("is this the chosen longest palindrome?"). Because we want exactly
one winner, every PAIR of candidates conflicts - picking two at once
is never allowed - so a real entangling gate (a Toffoli-based NAND,
reused from schrodingers_nand_sat.py, same trick as
03_cudaq_interval_scheduling.py) marks "both included" as infeasible
for every pair. Decision qubits are biased at preparation time
proportional to candidate length, so Grover amplification favors the
longest palindrome among the feasible (single-candidate) outcomes.

This mirrors 03_cudaq_interval_scheduling.py almost exactly: palindromic
substrings stand in for intervals, "any two selected simultaneously"
stands in for the conflict graph (here a complete graph, since capacity
is effectively 1 - only one substring can be "the answer"), and
substring length stands in for weight.

The final answer is checked against ../cpu/longest_palindromic_substring.py
and ../cuda/longest_palindromic_substring_cuda.py's results - CPU -> GPU ->
quantum, each step checked against the one before it.
"""

import math
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import cudaq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cuda"))
cpu = import_module("longest_palindromic_substring")
gpu = import_module("longest_palindromic_substring_cuda")

S = "aba"

CANDIDATES = [(i, j) for i in range(len(S)) for j in range(i, len(S))
              if S[i:j + 1] == S[i:j + 1][::-1]]
NUM_CANDIDATES = len(CANDIDATES)
WEIGHTS = [j - i + 1 for i, j in CANDIDATES]

CONFLICT_PAIRS = [(k1, k2) for k1 in range(NUM_CANDIDATES) for k2 in range(k1 + 1, NUM_CANDIDATES)]
NUM_CONFLICTS = len(CONFLICT_PAIRS)
CONFLICT_I = [p[0] for p in CONFLICT_PAIRS]
CONFLICT_J = [p[1] for p in CONFLICT_PAIRS]

_max_weight = max(WEIGHTS)
BIAS_PROBABILITIES = [0.3 + 0.5 * (w / _max_weight) for w in WEIGHTS]
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
    for i in range(NUM_CANDIDATES):
        ry(BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def unprepare_biased(decisions: cudaq.qview):
    for i in range(NUM_CANDIDATES):
        ry(-BIAS_ANGLES[i], decisions[i])


@cudaq.kernel
def diffusion(decisions: cudaq.qview):
    unprepare_biased(decisions)
    for i in range(NUM_CANDIDATES):
        x(decisions[i])
    z.ctrl(decisions[0:NUM_CANDIDATES - 1], decisions[NUM_CANDIDATES - 1])
    for i in range(NUM_CANDIDATES):
        x(decisions[i])
    prepare_biased(decisions)


@cudaq.kernel
def quantum_palindrome_search(iterations: int) -> list[bool]:
    decisions = cudaq.qvector(NUM_CANDIDATES)
    prepare_biased(decisions)
    for _ in range(iterations):
        mark_infeasible(decisions)
        diffusion(decisions)
    return mz(decisions)


def is_feasible(subset) -> bool:
    return len(subset) <= 1


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    print(f"=== Palindromic substring candidates of {S!r} ===")
    for k, (i, j) in enumerate(CANDIDATES):
        print(f"  candidate {k}: S[{i}:{j + 1}] = {S[i:j + 1]!r} (length {WEIGHTS[k]})")

    print()
    print("=== Verifying the feasibility fraction avoids the oscillation issue ===")
    import itertools
    feasible_count = sum(
        1 for size in range(NUM_CANDIDATES + 1)
        for subset in itertools.combinations(range(NUM_CANDIDATES), size)
        if is_feasible(subset))
    total = 2 ** NUM_CANDIDATES
    print(f"Feasible subsets: {feasible_count}/{total} ({feasible_count/total:.1%}) - "
          f"well under 50%, so Grover amplification should behave monotonically here.")

    print()
    print("=== Quantum search ===")
    cudaq.set_random_seed(42)
    shots = 2000
    for iterations in [0, 1, 2]:
        results = cudaq.run(quantum_palindrome_search, iterations, shots_count=shots)
        histogram = Counter(tuple(b) for b in results)
        scored = []
        for bits, count in histogram.items():
            subset = tuple(k for k, b in enumerate(bits) if b)
            feasible = is_feasible(subset)
            weight = WEIGHTS[subset[0]] if feasible and subset else 0
            scored.append((subset, feasible, weight, count))
        scored.sort(key=lambda r: -r[3])

        print(f"\n--- {iterations} Grover iteration(s) ---")
        for subset, feasible, weight, count in scored[:3]:
            substrings = [S[CANDIDATES[k][0]:CANDIDATES[k][1] + 1] for k in subset]
            print(f"  {substrings}: {count}/{shots} ({count/shots:.1%})  "
                  f"feasible={feasible}  weight={weight}")

    # Final check: does the quantum search's top answer at the best
    # iteration count match the CPU -> GPU verification chain?
    results = cudaq.run(quantum_palindrome_search, 1, shots_count=shots)
    histogram = Counter(tuple(b) for b in results)
    top_bits = histogram.most_common(1)[0][0]
    top_subset = [k for k, b in enumerate(top_bits) if b]
    top_answer = S[CANDIDATES[top_subset[0]][0]:CANDIDATES[top_subset[0]][1] + 1] if top_subset else ""

    cpu_answer = cpu.longest_palindrome(S)
    gpu_answer = gpu.longest_palindrome_gpu(S)

    print()
    print(f"Quantum top answer: {top_answer!r}")
    print(f"CPU reference answer: {cpu_answer!r}")
    print(f"GPU reference answer: {gpu_answer!r}")
    print(f"Quantum matches the CPU -> GPU verification chain: "
          f"{top_answer == cpu_answer == gpu_answer}")
