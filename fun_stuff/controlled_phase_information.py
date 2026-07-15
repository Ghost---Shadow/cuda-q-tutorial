"""
Controlled Phase Information
==============================

The exact circuit requested for this demo:

    x.ctrl(a, c)
    z.ctrl(b, c)

    mz(c)

a and b are both put in superposition (h(a); h(b)) before these 2
gates touch c. The question this script answers precisely: does
measuring c reveal information dictated by BOTH a and b, or just one
of them?

x.ctrl(a, c) is a CNOT - a REAL bit flip on c, conditioned on a. Its
effect on c's measured value is unambiguous and large.

z.ctrl(b, c) is a CZ - it only multiplies the |a=1,b=1,c=1> branch's
amplitude by -1. A minus sign in front of a computational-basis term
changes nothing about |amplitude|^2, which is all a plain mz() ever
reads. So the direct, verifiable answer is: **a fully dictates c's
measured value; b's action is invisible to mz(c) - it writes a
relative PHASE, not a bit flip.**

That could be where the story ends, except the phase is real, not
nothing: this script goes on to show, with an exact statevector
(cudaq.get_state(), no sampling noise) and then live sampling, that
b's contribution is recoverable - not by looking at c differently, but
by measuring b itself through a different lens (h(b) before mz(b),
same "different lens" trick as schrodingers_detective.py's quantum
eraser). Doing that snaps a, b, and c into a genuine 3-way GHZ-style
correlation that is completely absent when b is measured the obvious
way - concrete proof the phase was real information all along, just
stored somewhere a direct measurement of c can't reach.
"""

import cudaq

BASIS_LABELS = [format(i, "03b") for i in range(8)]


@cudaq.kernel
def prepare_and_gate() -> None:
    """h(a); h(b); x.ctrl(a,c); z.ctrl(b,c) - no measurement, so
    get_state() can read the exact amplitudes before anything
    collapses (same reason bell_state_unmeasured() in
    03_executing_kernels.py has no mz call)."""
    q = cudaq.qvector(3)
    a, b, c = q[0], q[1], q[2]
    h(a)
    h(b)
    x.ctrl(a, c)
    z.ctrl(b, c)


@cudaq.kernel
def measure_c_the_obvious_way() -> list[bool]:
    """The requested circuit, measured exactly as written: mz(c) (plus
    a and b, so we can check what actually correlates with what)."""
    q = cudaq.qvector(3)
    a, b, c = q[0], q[1], q[2]
    h(a)
    h(b)
    x.ctrl(a, c)
    z.ctrl(b, c)
    return mz(q)


@cudaq.kernel
def same_circuit_without_the_cz() -> list[bool]:
    """Control experiment: identical circuit with z.ctrl(b,c) deleted.
    If measurement statistics come out the same as the real circuit,
    that's direct proof the CZ gate is invisible to mz() here."""
    q = cudaq.qvector(3)
    a, b, c = q[0], q[1], q[2]
    h(a)
    h(b)
    x.ctrl(a, c)
    return mz(q)


@cudaq.kernel
def measure_b_through_a_different_lens() -> list[bool]:
    """Identical circuit, but b is measured through a different lens -
    h(b) right before mz(b), same mechanism as
    schrodingers_detective.py's view_through_different_lens(). a and c
    are still measured the obvious way."""
    q = cudaq.qvector(3)
    a, b, c = q[0], q[1], q[2]
    h(a)
    h(b)
    x.ctrl(a, c)
    z.ctrl(b, c)
    h(b)
    return mz(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== Part 1: exact statevector (no sampling noise) ===")
    state = cudaq.get_state(prepare_and_gate)
    print("Nonzero amplitudes, basis order (a,b,c):")
    for i in range(8):
        amp = state[i]
        if abs(amp) > 1e-9:
            print(f"  |{BASIS_LABELS[i]}>  amplitude = {amp.real:+.4f}")
    print("Expected by hand: |000>=+0.5, |010>=+0.5, |101>=+0.5, |111>=-0.5")
    print("Only the (a=1,b=1,c=1) branch flips sign - that's the CZ's entire")
    print("effect. Every amplitude has the same MAGNITUDE (0.5), so every")
    print("|amplitude|^2 = 0.25 - the CZ changes zero measurement probabilities.")

    print()
    print("=== Part 2: does mz(c) actually depend on b? ===")
    shots = 20000
    results = cudaq.run(measure_c_the_obvious_way, shots_count=shots)
    for a_val in (False, True):
        for b_val in (False, True):
            matching = [r for r in results if r[0] == a_val and r[1] == b_val]
            p_c1 = sum(1 for r in matching if r[2]) / len(matching) if matching else float("nan")
            print(f"  a={int(a_val)} b={int(b_val)}  ->  P(c=1) = {p_c1:.3f}  (n={len(matching)})")
    print("P(c=1) only ever depends on a (1.000 when a=1, 0.000 when a=0) -")
    print("flipping b between the two rows for a fixed a changes nothing.")

    print()
    print("=== Part 3: control experiment - delete the CZ, compare stats ===")
    results_no_cz = cudaq.run(same_circuit_without_the_cz, shots_count=shots)
    with_cz_counts = {label: 0 for label in BASIS_LABELS}
    without_cz_counts = {label: 0 for label in BASIS_LABELS}
    for r in results:
        with_cz_counts["".join(str(int(x)) for x in r)] += 1
    for r in results_no_cz:
        without_cz_counts["".join(str(int(x)) for x in r)] += 1
    print(f"  {'outcome (a,b,c)':<18}{'with z.ctrl':<14}{'without z.ctrl':<16}")
    for label in BASIS_LABELS:
        p_with = with_cz_counts[label] / shots
        p_without = without_cz_counts[label] / shots
        if p_with > 0 or p_without > 0:
            print(f"  {label:<18}{p_with:<14.3f}{p_without:<16.3f}")
    print("Same 4 outcomes, same ~25% each, whether or not z.ctrl(b,c) ever")
    print("ran - direct confirmation the gate is invisible to this measurement.")

    print()
    print("=== Part 4: recovering b's contribution through a different lens ===")
    lens_results = cudaq.run(measure_b_through_a_different_lens, shots_count=shots)
    lens_counts = {label: 0 for label in BASIS_LABELS}
    for r in lens_results:
        lens_counts["".join(str(int(x)) for x in r)] += 1
    print("Same circuit, only change: h(b) applied right before mz(b).")
    for label in BASIS_LABELS:
        p = lens_counts[label] / shots
        if p > 0.01:
            print(f"  {label}: {p:.3f}")
    print("Only |000> and |111> ever show up now, each ~50% - a, b, and c")
    print("snap into a perfect 3-way GHZ correlation. Measured the obvious")
    print("way (Part 2), b looked uncorrelated with (a,c); through this lens,")
    print("b's outcome perfectly predicts BOTH a and c. Nothing about the")
    print("circuit before mz() changed - only the question asked of b did.")

    print()
    print("=== Honest conclusion ===")
    print("Directly answering the question: a dictates c's measured value")
    print("completely (Parts 1-3, exact and statistical agreement, zero")
    print("dependence on b). b does NOT dictate c's value in any sense a")
    print("plain mz(c) can see - z.ctrl only ever writes a relative phase,")
    print("and phase alone is provably invisible to computational-basis")
    print("measurement (Part 3's identical statistics with the CZ deleted")
    print("prove this, they don't just suggest it). But 'invisible to mz(c)'")
    print("is not the same as 'not real information' - Part 4 shows the same")
    print("unmodified circuit, viewed through a different lens on b alone,")
    print("reveals a genuine, exact 3-way correlation between a, b, and c that")
    print("was there the whole time, just encoded as phase rather than as a")
    print("bit flip.")
