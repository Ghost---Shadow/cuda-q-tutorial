"""
Tutorial 9: Shor's Algorithm

Rewritten by hand from NVIDIA's official CUDA-Q "Shor's Algorithm"
tutorial:
https://nvidia.github.io/cuda-quantum/latest/applications/python/shors.html

Shor's algorithm factors an integer N by reducing factoring to a
different problem quantum computers are good at: order-finding. Pick a
random `a` coprime to N, and find the smallest r such that
a^r = 1 (mod N) - that r is called the "multiplicative order" of a.
Once you have r, plain classical arithmetic (no quantum computer
needed) usually recovers a factor of N via gcd(a^(r/2) +- 1, N).

Classically, order-finding is about as hard as factoring itself - you'd
have to try exponents one at a time. The quantum part below finds r
using quantum phase estimation (QPE) in a number of steps that scales
polynomially with the size of N, instead of exponentially.

This example factors N=15 using a=7 (order-4 example: 7^4 = 2401 =
160*15 + 1). Like the official tutorial, this uses a hardcoded,
by-hand-constructed modular multiplication circuit for this specific
(a, N) rather than a general-purpose modular-arithmetic circuit - a
real, size-72*L^3-gate general circuit (L = bit-length of N) is
extreme overkill for demonstrating the algorithm on toy-sized numbers
that a classical computer could factor by inspection anyway.

Instead of NVIDIA's specific SWAP-network circuit for a=5, N=21, this
version builds the modular-multiplication oracle a different (and
arguably more transparent) way: compute the "multiply by a^(2^k) mod N"
permutation as an ordinary classical function, turn it into a unitary
permutation matrix, and register it with CUDA-Q as a custom gate via
cudaq.register_operation(). This sidesteps hand-designing a reversible
arithmetic circuit out of Toffolis and swaps, while still giving QPE a
real, verifiable unitary operator to control.
"""

import math
from collections import Counter
from fractions import Fraction

import numpy as np
import cudaq

N = 15                # the number we're factoring
A = 7                  # a coprime to N; 7^4 = 1 (mod 15), so order r=4
WORK_QUBITS = 4        # ceil(log2(N)) = 4: work register holds values 0..15
CONTROL_QUBITS = 4     # phase-estimation precision register


def build_mult_mod_permutation(multiplier: int, modulus: int, n_qubits: int):
    """Turns "multiply by `multiplier`, mod `modulus`" into a unitary
    permutation matrix over n_qubits worth of basis states.

    For x < modulus, maps |x> -> |(multiplier*x) mod modulus>. This is
    reversible (since gcd(multiplier, modulus) = 1, multiplication mod
    modulus is a bijection on {0, ..., modulus-1}) and therefore a
    valid quantum operation. For "padding" states x >= modulus (the
    register has more computational basis states than N, since it's a
    fixed number of qubits), we map x to itself - keeping the whole
    thing a permutation (and therefore unitary) even though those
    states are never actually reached by the algorithm.
    """
    dim = 2 ** n_qubits
    matrix = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        y = (multiplier * x) % modulus if x < modulus else x
        matrix[y, x] = 1.0
    return matrix


# Register one custom gate per control qubit, each implementing
# multiplication by a^(2^k) mod N directly - this is the standard
# "precompute the powers classically" trick: rather than applying the
# same "multiply by a" gate 2^k times in a row, we compute what
# multiplying by a 2^k times in a row amounts to (mod N) as one classical
# modular-exponentiation, and build a single gate for it.
for k in range(CONTROL_QUBITS):
    multiplier = pow(A, 2 ** k, N)
    cudaq.register_operation(
        f"mult_pow{k}",
        build_mult_mod_permutation(multiplier, N, WORK_QUBITS).flatten())


@cudaq.kernel
def inverse_qft(qubits: cudaq.qview, n: int):
    """The inverse Quantum Fourier Transform - converts a state whose
    amplitudes encode a phase (like the one phase estimation produces)
    into a computational-basis state that, measured, gives an integer
    approximation of that phase.
    """
    for i in range(n // 2):
        swap(qubits[i], qubits[n - 1 - i])
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, i, -1):
            angle = -math.pi / (2 ** (j - i))
            r1.ctrl(angle, qubits[j], qubits[i])
        h(qubits[i])


@cudaq.kernel
def order_finding_circuit() -> list[bool]:
    """Quantum phase estimation applied to the "multiply by a mod N"
    operator, to find its order r.

    Phase estimation works by preparing the control register in an
    equal superposition, using each control qubit to conditionally
    apply a different power of the operator being studied (kicking
    back a phase onto the control register proportional to that
    power), and then running an inverse QFT to read that phase out as
    an ordinary integer.

    Control qubit 0 must control the *largest* power (2^(t-1)) so it
    ends up as the most-significant measured bit once the inverse QFT
    (which treats index 0 as MSB) runs - this took some empirical
    verification against the QFT's known analytic output to get right.
    """
    control = cudaq.qvector(CONTROL_QUBITS)
    work = cudaq.qvector(WORK_QUBITS)

    # Start the work register at |1> - order-finding studies how
    # repeatedly multiplying 1 by `a` (mod N) cycles back to 1.
    x(work[WORK_QUBITS - 1])

    for i in range(CONTROL_QUBITS):
        h(control[i])

    mult_pow3.ctrl(control[0], work[0], work[1], work[2], work[3])
    mult_pow2.ctrl(control[1], work[0], work[1], work[2], work[3])
    mult_pow1.ctrl(control[2], work[0], work[1], work[2], work[3])
    mult_pow0.ctrl(control[3], work[0], work[1], work[2], work[3])

    inverse_qft(control, CONTROL_QUBITS)

    return mz(control)


def recover_order(measured_value: int, register_size: int, modulus: int, base: int):
    """Classical post-processing: turns one QPE measurement into a
    candidate order r, using a continued-fraction approximation.

    QPE gives us measured_value / 2^register_size, an approximation of
    s/r for some unknown integers s and r (s < r <= modulus). The
    continued-fraction expansion (here, Python's
    Fraction.limit_denominator) finds the simplest fraction close to
    that value - denominator capped at `modulus`, since we know r < N.
    """
    phase = Fraction(measured_value, 2 ** register_size)
    candidate_r = phase.limit_denominator(modulus).denominator
    if candidate_r == 0:
        return None
    # The continued fraction can return a divisor of the true order
    # (e.g. if s and r shared a common factor), so check small
    # multiples of the candidate until one actually satisfies
    # a^r = 1 (mod N).
    for multiple in range(1, modulus + 1):
        candidate = candidate_r * multiple
        if candidate > 0 and pow(base, candidate, modulus) == 1:
            return candidate
    return None


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print(f"Factoring N={N} using a={A}")
    print(f"(precomputed powers a^(2^k) mod N: "
          f"{[pow(A, 2 ** k, N) for k in range(CONTROL_QUBITS)]})")

    print()
    print("=== Running quantum phase estimation ===")
    shots = 2000
    raw_results = cudaq.run(order_finding_circuit, shots_count=shots)
    histogram = Counter(
        int("".join("1" if bit else "0" for bit in bits), 2)
        for bits in raw_results)
    for value, count in sorted(histogram.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  measured {value:2d}/{2 ** CONTROL_QUBITS} "
              f"= {value / 2 ** CONTROL_QUBITS:.4f}   (x{count})")
    print("Expected: peaks concentrated at multiples of "
          f"{2 ** CONTROL_QUBITS}/{4} = {2 ** CONTROL_QUBITS // 4} "
          "(i.e. 0, 4, 8, 12), each ~25% of shots - these are exactly "
          "s/4 for s=0,1,2,3, revealing the true order r=4.")

    print()
    print("=== Classical post-processing: recovering N's factors ===")
    found_factors = None
    for value, _ in sorted(histogram.items(), key=lambda kv: -kv[1]):
        if value == 0:
            continue  # s=0 carries no information about r
        order = recover_order(value, CONTROL_QUBITS, N, A)
        if order is None or order % 2 != 0:
            continue
        half_power = pow(A, order // 2, N)
        if half_power == N - 1:
            continue  # this measurement can't yield a nontrivial factor
        factor_1 = math.gcd(half_power - 1, N)
        factor_2 = math.gcd(half_power + 1, N)
        if factor_1 not in (1, N):
            print(f"From measurement {value}: recovered order r={order}, "
                  f"factors {factor_1} x {factor_2} = {factor_1 * factor_2}")
            found_factors = (factor_1, factor_2)
            break

    print()
    if found_factors:
        print(f"N={N} = {found_factors[0]} x {found_factors[1]}")
    else:
        print("No nontrivial factors found from this measurement set - "
              "in practice you'd retry with a different `a`.")
