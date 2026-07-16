"""
A<->B (Bell pair), then A->C, B->D, then C->E, D->F. Is E-F correlated
despite never touching each other? Yes - it's just a 6-qubit GHZ state
built via two branches instead of one CNOT chain. Not entanglement
swapping (that needs a joint measurement between two independently
prepared pairs). Not a no-cloning workaround either: entangling erases
A's own local state into a maximally-mixed marginal - not a copy of
anything, verified below.
"""

import math
from collections import Counter

import cudaq

ROOT_ANGLE_RADIANS = math.radians(30)
ROOT_PROBABILITY = math.sin(ROOT_ANGLE_RADIANS / 2) ** 2


@cudaq.kernel
def correlate(control: cudaq.qubit, target: cudaq.qubit):
    x.ctrl(control, target)


@cudaq.kernel
def bell_pair_at_angles(theta_a: float, theta_b: float) -> list[bool]:
    """CHSH test on a plain, untouched Bell pair: measure A at Bloch
    angle theta_a, B at theta_b (ry(-theta) un-rotates that axis back
    to Z before measuring)."""
    q = cudaq.qvector(2)
    h(q[0])
    correlate(q[0], q[1])
    ry(-theta_a, q[0])
    ry(-theta_b, q[1])
    return mz(q)


@cudaq.kernel
def branch_pair_at_angles(theta_a: float, theta_e: float) -> list[bool]:
    """Same CHSH test, but on the DERIVED A-E pair from the 6-qubit
    chain instead of an untouched Bell pair."""
    q = cudaq.qvector(6)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    ry(-theta_a, q[0])
    ry(-theta_e, q[4])
    return [mz(q[0]), mz(q[4])]


@cudaq.kernel
def ef_pair_at_angles(theta_e: float, theta_f: float) -> list[bool]:
    """CHSH test on E-F specifically - the pair with NO direct gate
    between them at all, only chained correlation via C and D."""
    q = cudaq.qvector(6)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    ry(-theta_e, q[4])
    ry(-theta_f, q[5])
    return [mz(q[4]), mz(q[5])]


def chsh_s(run_kernel, shots: int) -> float:
    """Standard CHSH: angles 0/90/45/135 degrees, S = E(a,b) - E(a,b')
    + E(a',b) + E(a',b'). |S|<=2 is the classical bound; a real Bell
    pair reaches 2*sqrt(2)=2.828."""
    a, a_ = 0.0, math.radians(90)
    b, b_ = math.radians(45), math.radians(135)

    def e(theta_a, theta_b):
        results = run_kernel(theta_a, theta_b, shots_count=shots)
        agree = sum(1 for r in results if r[0] == r[1])
        return (2 * agree - shots) / shots  # +1 if agree, -1 if disagree

    return e(a, b) - e(a, b_) + e(a_, b) + e(a_, b_)


@cudaq.kernel
def chain_biased() -> list[bool]:
    q = cudaq.qvector(6)
    ry(ROOT_ANGLE_RADIANS, q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    return mz(q)


@cudaq.kernel
def chain_control() -> list[bool]:
    """Same shape, but A/B start independent - rules out "any circuit
    shaped like this correlates C-D/E-F" as the explanation."""
    q = cudaq.qvector(6)
    ry(ROOT_ANGLE_RADIANS, q[0])
    ry(ROOT_ANGLE_RADIANS, q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    return mz(q)


@cudaq.kernel
def a_alone_x() -> bool:
    """A, untouched, measured in the X basis - a real pure |+> state."""
    a = cudaq.qubit()
    h(a)
    h(a)
    return mz(a)


@cudaq.kernel
def a_after_x() -> bool:
    """The same A, after entangling with B..F, measured in X alone."""
    q = cudaq.qvector(6)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    h(q[0])
    return mz(q[0])


@cudaq.kernel
def e_after_x() -> bool:
    """E alone, measured in X, after the same chain."""
    q = cudaq.qvector(6)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    correlate(q[2], q[4])
    correlate(q[3], q[5])
    h(q[4])
    return mz(q[4])


@cudaq.kernel
def classical_coin_x(prepared_as_one: bool) -> bool:
    """No entanglement at all - a definite |0>/|1>, measured in X."""
    q = cudaq.qubit()
    if prepared_as_one:
        x(q)
    h(q)
    return mz(q)


@cudaq.kernel
def clone_z() -> list[bool]:
    """Try to "clone" the Bell pair (a,b) into (a2,b2) via CNOT-copy."""
    q = cudaq.qvector(4)
    ry(ROOT_ANGLE_RADIANS, q[0])
    correlate(q[0], q[1])   # original pair
    correlate(q[0], q[2])   # "clone" a -> a2
    correlate(q[1], q[3])   # "clone" b -> b2
    return mz(q)


@cudaq.kernel
def original_pair_x() -> list[bool]:
    """(a,b) alone in the X basis, after the clone attempt ran."""
    q = cudaq.qvector(4)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    h(q[0])
    h(q[1])
    return [mz(q[0]), mz(q[1])]


@cudaq.kernel
def untouched_pair_x() -> list[bool]:
    """A real, standalone Bell pair - baseline for original_pair_x()."""
    q = cudaq.qvector(2)
    h(q[0])
    correlate(q[0], q[1])
    h(q[0])
    h(q[1])
    return [mz(q[0]), mz(q[1])]


@cudaq.kernel
def clone_pair_x() -> list[bool]:
    """(a2,b2) alone in the X basis - does the "clone" work standalone?"""
    q = cudaq.qvector(4)
    h(q[0])
    correlate(q[0], q[1])
    correlate(q[0], q[2])
    correlate(q[1], q[3])
    h(q[2])
    h(q[3])
    return [mz(q[2]), mz(q[3])]


def agree_rate(results: list) -> float:
    return sum(1 for r in results if r[0] == r[1]) / len(results)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)
    shots = 2000

    biased = cudaq.run(chain_biased, shots_count=shots)
    control = cudaq.run(chain_control, shots_count=shots)

    outcomes = Counter(tuple(r) for r in biased)
    only_agree = all(o in ((False,) * 6, (True,) * 6) for o in outcomes)
    p_ef_biased = sum(1 for r in biased if r[4] and r[5]) / shots
    p_ef_control = sum(1 for r in control if r[4] and r[5]) / shots
    p_e_biased = sum(1 for r in biased if r[4]) / shots

    print(f"Root bias: predicted P(1) = {ROOT_PROBABILITY:.1%}")
    print(f"Chain (entangled root): only-agree outcomes = {only_agree}, "
          f"P(E=1 AND F=1) = {p_ef_biased:.3f} (independent guess: {p_e_biased**2:.3f})")
    print(f"Chain (independent root, control): P(E=1 AND F=1) = {p_ef_control:.3f} "
          f"(independent guess: {p_e_biased**2:.3f})")
    print("-> E-F correlated only when A-B started entangled. Success.")

    p_ae_match = sum(1 for r in biased if r[0] == r[4]) / shots
    p_match_if_independent_clone = p_e_biased**2 + (1 - p_e_biased)**2
    print(f"\nPer-shot A==E match rate: {p_ae_match:.3f} "
          f"(if E were an INDEPENDENT draw from the same {ROOT_PROBABILITY:.1%} "
          f"distribution: {p_match_if_independent_clone:.3f})")
    print("-> A programmer's clone would agree with A only by chance, on this many")
    print("   shots. This agrees EVERY shot: A, E aren't two draws from the same")
    print("   distribution - they're the SAME single draw, read from 2 places.")

    a1 = [r for r in biased if r[0]]
    a0 = [r for r in biased if not r[0]]
    p_e_given_a1 = sum(1 for r in a1 if r[4]) / len(a1)
    p_e_given_a0 = sum(1 for r in a0 if r[4]) / len(a0)
    print(f"\nDoes measuring A collapse E? P(E=1) unconditional = {p_e_biased:.3f}; "
          f"P(E=1 | A=1) = {p_e_given_a1:.3f}; P(E=1 | A=0) = {p_e_given_a0:.3f}")
    print("-> E's own statistics FLIP from ~6.7% to ~100% the instant A=1 is known.")
    print("   That's collapse: measuring A resolves E's uncertainty, not a")
    print("   'clone by value' that could be measured independently untouched.")

    print()
    p_a_alone = sum(cudaq.run(a_alone_x, shots_count=shots)) / shots
    p_a_after = sum(cudaq.run(a_after_x, shots_count=shots)) / shots
    p_e_after = sum(cudaq.run(e_after_x, shots_count=shots)) / shots
    p_coin = sum(
        cudaq.run(classical_coin_x, False, shots_count=shots // 2)
        + cudaq.run(classical_coin_x, True, shots_count=shots // 2)
    ) / shots
    print(f"No-cloning check (X basis, single qubit): A-alone={p_a_alone:.3f} "
          f"(pure), A-after={p_a_after:.3f}, E-after={p_e_after:.3f}, "
          f"classical-coin={p_coin:.3f}")
    print("-> entangling erases A's own state (0.0 -> 0.5), same as a coin flip.")
    print("   E never held an independent copy of A's original state.")

    print()
    p_clone_match = sum(1 for r in cudaq.run(clone_z, shots_count=shots)
                         if r[0] == r[2] and r[1] == r[3]) / shots
    p_untouched = agree_rate(cudaq.run(untouched_pair_x, shots_count=shots))
    p_original_after = agree_rate(cudaq.run(original_pair_x, shots_count=shots))
    p_clone_pair = agree_rate(cudaq.run(clone_pair_x, shots_count=shots))
    print(f"clone_gate() check: Z-basis match = {p_clone_match:.3f} (looks cloned)")
    print(f"X-basis agreement: untouched={p_untouched:.3f}, "
          f"original-after-clone={p_original_after:.3f}, clone-pair={p_clone_pair:.3f}")
    print("-> clone attempt disturbs the original and doesn't produce a working")
    print("   independent copy either. No-cloning holds; no custom gate fixes it.")

    print()
    s_bell = chsh_s(
        lambda ta, tb, shots_count: cudaq.run(bell_pair_at_angles, ta, tb, shots_count=shots_count),
        shots)
    s_branch = chsh_s(
        lambda ta, tb, shots_count: cudaq.run(branch_pair_at_angles, ta, tb, shots_count=shots_count),
        shots)
    s_ef = chsh_s(
        lambda ta, tb, shots_count: cudaq.run(ef_pair_at_angles, ta, tb, shots_count=shots_count),
        shots)
    print(f"CHSH S (untouched Bell pair, A-B): {s_bell:.3f}  "
          f"(classical limit 2.000, quantum max 2.828)")
    print(f"CHSH S (derived pair, A-E): {s_branch:.3f}")
    print(f"CHSH S (derived pair, E-F - no direct gate between them at all): {s_ef:.3f}")
    print("-> |S|>2 means NO pre-shared-secret / hidden-variable scheme could")
    print("   produce these correlations. A-B breaks the classical bound; the")
    print("   derived pairs don't, even E-F, the pair this whole file started")
    print("   with (correlated 100% of the time in Z, per-shot). Monogamy of")
    print("   entanglement: spreading correlation across more qubits weakens")
    print("   any single pair's own Bell violation, even while the FULL joint")
    print("   6-qubit state stays a genuine, non-classical GHZ state.")
