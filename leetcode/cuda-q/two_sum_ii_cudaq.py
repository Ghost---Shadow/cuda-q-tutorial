"""LeetCode 167: Two Sum II - Input Array Is Sorted - CUDA-Q

Same problem as ../cpu/two_sum_ii.py and ../cuda/two_sum_ii_cuda.py, but
neither two-pointer's sortedness exploit nor the GPU's brute-force pairwise
scan translate directly into a quantum speedup - so this does something
genuinely different instead of a token "quantumification": treat the
C(n,2) candidate pairs as an unstructured search space (deliberately
throwing away the sortedness, same honest tradeoff the CUDA version makes
by using brute force instead of two pointers) and use Grover's algorithm -
the exact oracle+diffusion pattern from
nvidia-cudaq-tutorials/08_grovers_algorithm.py, retargeted at a candidate
INDEX register instead of a hardcoded 4-qubit pattern - to amplify the one
index whose pair sums to target.

Honest scope note: Grover's gives O(sqrt(C(n,2))) = O(n) oracle calls here,
the SAME asymptotic order as the two-pointer method - this is not a case
where quantum search beats the best classical algorithm for this specific
problem (the sortedness makes O(n) two-pointer already optimal). It's
presented as an honest demonstration of the mechanism - "which candidate
index satisfies a checkable condition" is exactly Grover's home turf - not
a speedup claim.

Worked on the primary example (numbers=[2,7,11,15], target=9) - the other
2 official examples are still exercised end-to-end in
../cpu/two_sum_ii_test.py and ../cuda/two_sum_ii_cuda_test.py, since a
different input changes the qubit count (candidate index width depends on
n), same reason longest_palindromic_substring_cudaq.py picks one fixed
string rather than looping over every test case.
"""

import math
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

import cudaq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cpu"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cuda"))
cpu = import_module("two_sum_ii")
gpu = import_module("two_sum_ii_cuda")

NUMBERS = [2, 7, 11, 15]
TARGET = 9
N = len(NUMBERS)

CANDIDATES = [(i, j) for i in range(N) for j in range(i + 1, N)]
NUM_CANDIDATES = len(CANDIDATES)
INDEX_WIDTH = max(1, (NUM_CANDIDATES - 1).bit_length())  # bits needed to address every candidate


@cudaq.kernel
def mark_index(qubits: cudaq.qview, bit2: bool, bit1: bool, bit0: bool):
    """Phase-flips exactly the one 3-qubit basis state matching
    (bit2, bit1, bit0) - same temp-X-flip + multi-controlled-Z pattern as
    08_grovers_algorithm.py's mark_1001/mark_1111, retargeted at a
    runtime-supplied candidate index instead of a hardcoded one."""
    if not bit2:
        x(qubits[0])
    if not bit1:
        x(qubits[1])
    if not bit0:
        x(qubits[2])
    z.ctrl(qubits[0], qubits[1], qubits[2])
    if not bit0:
        x(qubits[2])
    if not bit1:
        x(qubits[1])
    if not bit2:
        x(qubits[0])


@cudaq.kernel
def diffusion(qubits: cudaq.qview):
    """Grover diffusion for a 3-qubit index register - identical structure
    to 08_grovers_algorithm.py's diffusion(), just 3 qubits."""
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    z.ctrl(qubits[0], qubits[1], qubits[2])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])


@cudaq.kernel
def two_sum_search(bit2: bool, bit1: bool, bit0: bool, iterations: int) -> list[bool]:
    q = cudaq.qvector(INDEX_WIDTH)
    h(q[0])
    h(q[1])
    h(q[2])
    for _ in range(iterations):
        mark_index(q, bit2, bit1, bit0)
        diffusion(q)
    return mz(q)


def correct_candidate_index() -> int:
    matches = [k for k, (i, j) in enumerate(CANDIDATES) if NUMBERS[i] + NUMBERS[j] == TARGET]
    assert len(matches) == 1, "problem guarantees exactly one solution"
    return matches[0]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(2)

    print(f"numbers={NUMBERS}, target={TARGET}")
    print(f"{NUM_CANDIDATES} candidate pairs, {INDEX_WIDTH}-qubit index "
          f"(search space {2 ** INDEX_WIDTH})")
    for k, (i, j) in enumerate(CANDIDATES):
        print(f"  index {k}: numbers[{i}]+numbers[{j}] = {NUMBERS[i]}+{NUMBERS[j]} = {NUMBERS[i]+NUMBERS[j]}")

    correct_index = correct_candidate_index()
    bit2, bit1, bit0 = bool(correct_index & 4), bool(correct_index & 2), bool(correct_index & 1)
    iterations = math.floor((math.pi / 4) * math.sqrt(2 ** INDEX_WIDTH))
    print(f"\ncorrect candidate index: {correct_index}  optimal Grover iterations: {iterations}")

    shots = 2000
    results = cudaq.run(two_sum_search, bit2, bit1, bit0, iterations, shots_count=shots)
    counts = Counter(
        (4 if r[0] else 0) + (2 if r[1] else 0) + (1 if r[2] else 0)
        for r in results
    )
    hit_rate = counts[correct_index] / shots
    print(f"hit rate on the correct index: {hit_rate:.1%}  "
          f"(top results: {counts.most_common(3)})")

    top_index = counts.most_common(1)[0][0]
    i, j = CANDIDATES[top_index]
    quantum_answer = [i + 1, j + 1]
    cpu_answer = cpu.two_sum(NUMBERS, TARGET)
    gpu_answer = gpu.two_sum_gpu(NUMBERS, TARGET)

    print(f"\nQuantum top answer: {quantum_answer}")
    print(f"CPU reference answer: {cpu_answer}")
    print(f"GPU reference answer: {gpu_answer}")
    print(f"Quantum matches the CPU -> GPU verification chain: "
          f"{quantum_answer == cpu_answer == gpu_answer}")
