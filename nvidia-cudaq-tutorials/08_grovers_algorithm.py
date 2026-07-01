"""
Tutorial 8: Grover's Algorithm

Rewritten by hand from NVIDIA's official CUDA-Q "Grover's Algorithm"
tutorial:
https://nvidia.github.io/cuda-quantum/latest/applications/python/grovers.html

The problem: search an unsorted list of N=16 items (encoded as every
possible 4-qubit bitstring) for a small set of "marked" items, using
only a black-box oracle that can recognize a marked item when it sees
one. Classically this takes O(N) lookups on average. Grover's algorithm
does it in O(sqrt(N)) - here, 2 rounds instead of an average of ~8
sequential guesses.

This example marks two 4-qubit bitstrings: 1001 (decimal 9) and 1111
(decimal 15).

Grover's algorithm alternates two steps, each a "reflection":
  1. The oracle flips the sign (phase) of marked states only, leaving
     everything else untouched. This doesn't change any measurement
     probabilities by itself (probabilities come from amplitude
     *magnitudes*, and flipping a sign doesn't change a magnitude) -
     but it sets up interference for the next step.
  2. The diffusion operator reflects every amplitude about the average
     amplitude. Combined with step 1, this has the effect of shrinking
     the un-marked amplitudes and growing the marked ones, step by
     step, until measuring gives a marked state with high probability.
"""

import math

import cudaq


@cudaq.kernel
def mark_1111(qubits: cudaq.qview):
    """Flips the phase of |1111> only, via a 4-qubit-controlled Z.

    Z's phase flip only has an effect on |1>, so multi-controlling a Z
    gate on qubits[3] by qubits[0..2] means: only when *all four*
    qubits read |1> does anything happen at all. Every other basis
    state passes through completely unaffected - exactly the "mark
    this one specific item" behavior an oracle needs.
    """
    z.ctrl(qubits[0], qubits[1], qubits[2], qubits[3])


@cudaq.kernel
def mark_1001(qubits: cudaq.qview):
    """Flips the phase of |1001> only.

    The 4-qubit-controlled Z above only fires on |1111>. To retarget
    it at |1001> instead, temporarily flip the two qubits that should
    read 0 in the target pattern (qubits 1 and 2) with X gates first -
    that turns "|1001>" into "|1111>" in a shifted frame, where the
    same controlled-Z fires - then flip those two qubits back with X
    again so the rest of the circuit sees the original basis.
    """
    x(qubits[1])
    x(qubits[2])
    z.ctrl(qubits[0], qubits[1], qubits[2], qubits[3])
    x(qubits[1])
    x(qubits[2])


@cudaq.kernel
def oracle(qubits: cudaq.qview):
    """Marks both target states by composing the two single-state
    oracles above - phase flips are independent per basis state, so
    marking two different states is just doing both in sequence."""
    mark_1001(qubits)
    mark_1111(qubits)


@cudaq.kernel
def diffusion(qubits: cudaq.qview):
    """Reflects the amplitude vector about the uniform-superposition
    average - the second half of one Grover iteration.

    The trick: H moves the uniform superposition to |0000>, X moves
    |0000> to |1111>, the multi-controlled Z flips the phase of
    (what's now) |1111> - i.e. of the *original* uniform superposition
    - and then X, H undo the basis change. Net effect: every amplitude
    is reflected about the average of all amplitudes, which is exactly
    the "inversion about the mean" that gives Grover's algorithm its
    quadratic speedup.
    """
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    h(qubits[3])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    x(qubits[3])
    z.ctrl(qubits[0], qubits[1], qubits[2], qubits[3])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    x(qubits[3])
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    h(qubits[3])


@cudaq.kernel
def grover_search(iterations: int):
    qubits = cudaq.qvector(4)
    # Start in an equal superposition of all 16 possible answers -
    # every bitstring is equally likely before we do anything else.
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    h(qubits[3])

    for _ in range(iterations):
        oracle(qubits)
        diffusion(qubits)

    mz(qubits)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    total_states = 16  # N = 2^4 possible answers
    marked_states = 2  # |1001> and |1111>

    # The optimal number of iterations for Grover's algorithm - too few
    # and the marked amplitudes haven't grown enough yet; too many and
    # you "overshoot" past the peak and start shrinking them again.
    optimal_iterations = math.floor(
        (math.pi / 4) * math.sqrt(total_states / marked_states)
    )
    print(f"Optimal iterations: {optimal_iterations}")

    print()
    print("=== Running Grover's algorithm ===")
    result = cudaq.sample(grover_search, optimal_iterations, shots_count=2000)
    print(result)

    print()
    print("Expected: '1001' and '1111' should each account for roughly")
    print("45-50% of the 2000 shots, with every other bitstring showing")
    print("up rarely if at all - versus 1/16 (~6%) each if we'd just")
    print("measured the initial uniform superposition with no Grover")
    print("iterations at all.")

    print()
    print("=== Effect of iteration count (searching for the sweet spot) ===")
    # Too few iterations under-rotates towards the marked states; too
    # many over-rotates past them and probability starts dropping again
    # - the classic Grover "overshoot" behavior.
    for iterations in range(0, 6):
        result = cudaq.sample(grover_search, iterations, shots_count=2000)
        hit_count = result.count("1001") + result.count("1111")
        print(
            f"iterations={iterations}: "
            f"{hit_count}/2000 shots landed on a marked state"
        )
