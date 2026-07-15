"""
Cyclic Dependency
===================

The exact circuit requested for this demo:

    h(a); h(b); h(c)
    x.ctrl(a, b)
    x.ctrl(b, c)
    x.ctrl(c, a)

Three qubits, each nudging the next one around a loop: a controls b,
b controls c, c controls a. On paper that "a depends on c which
depends on b which depends on a" reads like a circular reference a
spreadsheet would reject. A circuit can't reject it - gates apply in
one strict sequential order, so the loop has to be unrolled into
SOME order, and this script's whole point is to make that unrolling
visible: which edge closes the loop first changes the resulting
permutation, even though it's built from the identical 3 edges.

x.ctrl (unlike schrodingers_epidemic.py's ry.ctrl "soft nudge") is a
plain CNOT: on a computational-basis input, it's a fully deterministic
bit flip, not a probability shift. That makes this cycle a REVERSIBLE
CLASSICAL PERMUTATION of the 8 possible (a, b, c) bitstrings, wearing
quantum superposition on top - which gives 2 clean, exact (not just
statistical) things to check:

1. Forced classical inputs (no h()): every single shot should match a
   hand-computed classical truth table exactly, every time - zero
   sampling noise, because CNOTs on basis states are deterministic.
2. h(a); h(b); h(c) (the real superposition ask): each qubit is
   "cloned" into a frozen ancilla via CNOT right after h(), BEFORE the
   cycle touches it. That lets us recover, per shot, both the
   pre-cycle value AND the post-cycle value from ONE measurement, and
   check they still match the same truth table exactly, every shot -
   proof the deterministic cyclic dependency survives fully intact
   underneath the superposition, not just on average.

The honest surprise: because a permutation of a perfectly uniform
distribution is still perfectly uniform, P(a=1), P(b=1), P(c=1) come
out ~50% either way you order the loop - the reordering effect is
INVISIBLE in the marginals. You only see it by looking at the joint
(before, after) mapping, which is exactly what the ancilla clones are
for.
"""

from itertools import product

import cudaq


def classical_cycle_order1(a: bool, b: bool, c: bool):
    """x.ctrl(a,b); x.ctrl(b,c); x.ctrl(c,a) - the cycle-closing edge
    (c controls a) applied LAST."""
    b = b ^ a
    c = c ^ b
    a = a ^ c
    return a, b, c


def classical_cycle_order2(a: bool, b: bool, c: bool):
    """x.ctrl(c,a); x.ctrl(a,b); x.ctrl(b,c) - same 3 edges, but the
    cycle-closing edge (c controls a) applied FIRST instead."""
    a = a ^ c
    b = b ^ a
    c = c ^ b
    return a, b, c


@cudaq.kernel
def cycle_order1(use_superposition: bool, a_init: bool, b_init: bool, c_init: bool) -> list[bool]:
    q = cudaq.qvector(6)
    a, b, c, a0, b0, c0 = q[0], q[1], q[2], q[3], q[4], q[5]

    if use_superposition:
        h(a)
        h(b)
        h(c)
    else:
        if a_init:
            x(a)
        if b_init:
            x(b)
        if c_init:
            x(c)

    # Clone each qubit's pre-cycle value into a frozen ancilla, before
    # the cyclic dependency below gets a chance to change it.
    x.ctrl(a, a0)
    x.ctrl(b, b0)
    x.ctrl(c, c0)

    # The requested cycle, unrolled in this order: a->b, b->c, c->a.
    x.ctrl(a, b)
    x.ctrl(b, c)
    x.ctrl(c, a)

    return mz(q)


@cudaq.kernel
def cycle_order2(use_superposition: bool, a_init: bool, b_init: bool, c_init: bool) -> list[bool]:
    q = cudaq.qvector(6)
    a, b, c, a0, b0, c0 = q[0], q[1], q[2], q[3], q[4], q[5]

    if use_superposition:
        h(a)
        h(b)
        h(c)
    else:
        if a_init:
            x(a)
        if b_init:
            x(b)
        if c_init:
            x(c)

    x.ctrl(a, a0)
    x.ctrl(b, b0)
    x.ctrl(c, c0)

    # The SAME 3 edges, just with the cycle-closing one (c->a) moved
    # from last to first.
    x.ctrl(c, a)
    x.ctrl(a, b)
    x.ctrl(b, c)

    return mz(q)


def bits_label(a: bool, b: bool, c: bool) -> str:
    return f"({int(a)},{int(b)},{int(c)})"


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== Part 1: forced classical inputs - exact truth tables ===")
    print("No h() here - each (a,b,c) is set classically via x(), so CNOTs")
    print("acting on it are 100% deterministic. Every shot of every one of")
    print("the 8 inputs should land on the SAME output, no exceptions.")
    print()
    print(f"  {'input (a,b,c)':<16}{'order1 -> a->b->c->a':<24}{'order2 -> c->a->b->c':<24}{'differ?'}")
    order1_table = {}
    order2_table = {}
    for a_init, b_init, c_init in product([False, True], repeat=3):
        results1 = cudaq.run(cycle_order1, False, a_init, b_init, c_init, shots_count=100)
        results2 = cudaq.run(cycle_order2, False, a_init, b_init, c_init, shots_count=100)

        expected1 = classical_cycle_order1(a_init, b_init, c_init)
        expected2 = classical_cycle_order2(a_init, b_init, c_init)

        measured1 = {tuple(r[:3]) for r in results1}
        measured2 = {tuple(r[:3]) for r in results2}
        assert measured1 == {expected1}, f"order1 mismatch for {(a_init, b_init, c_init)}: got {measured1}"
        assert measured2 == {expected2}, f"order2 mismatch for {(a_init, b_init, c_init)}: got {measured2}"

        order1_table[(a_init, b_init, c_init)] = expected1
        order2_table[(a_init, b_init, c_init)] = expected2
        differ = "  <-- differs" if expected1 != expected2 else ""
        print(f"  {bits_label(a_init, b_init, c_init):<16}"
              f"{bits_label(*expected1):<24}"
              f"{bits_label(*expected2):<24}{differ}")

    num_differ = sum(1 for k in order1_table if order1_table[k] != order2_table[k])
    print()
    print(f"100/100 shots matched the predicted bit exactly for all 8 inputs,")
    print(f"in both orderings - zero sampling noise, as expected for pure CNOTs.")
    print(f"{num_differ} of 8 inputs produce a DIFFERENT output depending on which")
    print("edge closes the loop first - same 3 edges, genuinely different circuit.")

    print()
    print("=== Part 2: the actual ask - h(a); h(b); h(c) then the cycle ===")
    print("All 3 inputs now start in real superposition. Reading the frozen")
    print("ancilla clones alongside the post-cycle qubits lets us check the")
    print("SAME truth table holds shot-by-shot, even though (a,b,c) were never")
    print("classically fixed before the cycle ran.")
    shots = 5000

    for label, kernel, classical_fn in [
        ("order1 (a->b->c->a, cycle closes last)", cycle_order1, classical_cycle_order1),
        ("order2 (c->a->b->c, cycle closes first)", cycle_order2, classical_cycle_order2),
    ]:
        results = cudaq.run(kernel, True, False, False, False, shots_count=shots)
        mismatches = 0
        transition_counts = {}
        for r in results:
            a, b, c, a0, b0, c0 = r
            predicted = classical_fn(a0, b0, c0)
            if (a, b, c) != predicted:
                mismatches += 1
            transition_counts[(a0, b0, c0)] = transition_counts.get((a0, b0, c0), 0) + 1

        p_a = sum(1 for r in results if r[0]) / shots
        p_b = sum(1 for r in results if r[1]) / shots
        p_c = sum(1 for r in results if r[2]) / shots

        print(f"\n  --- {label} ---")
        print(f"  Shot-by-shot mismatches vs. the classical truth table: {mismatches}/{shots}")
        print(f"  Marginals after the cycle: P(a=1)={p_a:.3f}  P(b=1)={p_b:.3f}  P(c=1)={p_c:.3f}")
        print("  (~50% each, regardless of ordering - permuting a uniform")
        print("  distribution is still uniform, so the reordering effect is")
        print("  invisible here; it only shows up in the joint before/after map.")

    print()
    print("=== Honest conclusion ===")
    print("A cyclic dependency in the CIRCUIT DIAGRAM (a needs b, b needs c, c")
    print("needs a) is not a cyclic dependency at RUNTIME - gates still execute")
    print("in one fixed sequential order, so the loop is really just 3 CNOTs in")
    print("some order, and 'the cycle' is a drawing convention, not a real")
    print("simultaneous solve. What's genuinely demonstrated here: (1) the exact")
    print("same 3 edges produce a different permutation depending on which one")
    print("closes the loop, verified against a hand-computed truth table with")
    print("zero sampling noise (pure CNOTs are deterministic), and (2) that")
    print("deterministic mapping survives completely intact even when the 3")
    print("qubits start in genuine quantum superposition, checked shot-by-shot")
    print("via frozen ancilla clones taken before the cycle runs - while the")
    print("marginal statistics alone would have hidden the entire effect.")
