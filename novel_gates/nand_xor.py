"""
Starting misconception worth correcting: CNOT is NOT a NAND gate. CNOT
computes target ^= control - it's a reversible XOR, full stop. NAND can't
literally BE a quantum gate at all, because NAND isn't reversible: three
different input pairs - (0,0), (0,1), (1,0) - all map to the same output,
1. Every quantum gate is a unitary matrix, and unitary matrices are
bijections on basis states (info-preserving); a 2-inputs-to-1-output
function that collapses 3 cases into 1 can't be one, no matter how you
wire it up.

But there IS a real, standard way to make NAND reversible, the same trick
Toffoli uses for AND: give it an extra output qubit, preset that qubit to
|1>, and flip it conditioned on both inputs via a Toffoli gate. Since
1 XOR (a AND b) = NAND(a,b), that ancilla ends up holding exactly the NAND
of the two (untouched, still-readable) inputs - and the whole thing is
reversible (Toffoli is self-inverse, a and b pass through unchanged).

This file builds the classic 4-NAND XOR circuit -
  n1 = NAND(a,b);  n2 = NAND(a,n1);  n3 = NAND(b,n1);  xor = NAND(n2,n3)
- entirely out of that reversible-NAND primitive, and checks it against
both the classical truth table AND a native 2-CNOT XOR (target ^= a;
target ^= b) built in the same circuit - on fixed inputs, and on a genuine
superposition over all 4 (a,b) combinations, with no measurement until the
final readout.
"""

import cudaq
from collections import Counter


@cudaq.kernel
def reversible_nand(c1: cudaq.qubit, c2: cudaq.qubit, out: cudaq.qubit):
    """out (assumed fresh |0>) becomes NAND(c1, c2); c1, c2 pass through
    unchanged. Preset out=1, then Toffoli-flip it iff c1 AND c2 - so
    out = 1 XOR (c1 AND c2) = NOT(c1 AND c2). Reversible: c1, c2, and the
    Toffoli are all untouched/self-inverse, unlike classical NAND."""
    x(out)
    x.ctrl(c1, c2, out)


@cudaq.kernel
def xor_from_nand(a: cudaq.qubit, b: cudaq.qubit,
                   n1: cudaq.qubit, n2: cudaq.qubit, n3: cudaq.qubit,
                   out: cudaq.qubit):
    """The textbook 4-NAND XOR circuit, gate for gate, with each NAND
    replaced by reversible_nand(). n1, n2, n3 are intermediate NAND
    results (real, meaningful values, not hidden garbage - printed
    below)."""
    reversible_nand(a, b, n1)
    reversible_nand(a, n1, n2)
    reversible_nand(b, n1, n3)
    reversible_nand(n2, n3, out)


@cudaq.kernel
def compare_fixed(a_in: bool, b_in: bool) -> list[bool]:
    """Fixed (non-superposed) a, b - one classical case per run."""
    q = cudaq.qvector(7)
    a, b, n1, n2, n3, nand_xor_out, cnot_xor_out = q[0], q[1], q[2], q[3], q[4], q[5], q[6]
    if a_in:
        x(a)
    if b_in:
        x(b)

    xor_from_nand(a, b, n1, n2, n3, nand_xor_out)

    # Native XOR for comparison: 2 plain CNOTs into a fresh qubit.
    x.ctrl(a, cnot_xor_out)
    x.ctrl(b, cnot_xor_out)

    return mz(q)


@cudaq.kernel
def compare_superposed() -> list[bool]:
    """a, b both Hadamard'd - an equal superposition over all 4 input
    combinations, processed by the SAME unitary circuit. Shadow qubits
    snapshot a, b (a per-branch classical copy, not a clone of an unknown
    state) so each shot's sampled input can be checked against that same
    shot's output."""
    q = cudaq.qvector(9)
    a, b = q[0], q[1]
    n1, n2, n3, nand_xor_out, cnot_xor_out = q[2], q[3], q[4], q[5], q[6]
    shadow_a, shadow_b = q[7], q[8]

    h(a)
    h(b)
    x.ctrl(a, shadow_a)
    x.ctrl(b, shadow_b)

    xor_from_nand(a, b, n1, n2, n3, nand_xor_out)
    x.ctrl(a, cnot_xor_out)
    x.ctrl(b, cnot_xor_out)

    return mz(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(11)

    print("=== Truth table: reversible-NAND XOR vs native CNOT XOR vs classical a^b ===")
    print("a b | n1 n2 n3 | nand_xor cnot_xor expected")
    all_ok = True
    for a in (False, True):
        for b in (False, True):
            r = cudaq.run(compare_fixed, a, b, shots_count=1)[0]
            n1, n2, n3, nand_xor, cnot_xor = r[2], r[3], r[4], r[5], r[6]
            expected = a ^ b
            row_ok = (nand_xor == expected) and (cnot_xor == expected)
            all_ok = all_ok and row_ok
            print(f"{int(a)} {int(b)} | {int(n1)}  {int(n2)}  {int(n3)}  |    "
                  f"{int(nand_xor)}        {int(cnot_xor)}       {int(expected)}   "
                  f"{'OK' if row_ok else 'MISMATCH'}")
    print(f"Truth table: {'ALL OK' if all_ok else 'FAILURES PRESENT'}\n")

    print("=== Superposition: a, b both Hadamard'd, no mid-circuit measurement ===")
    shots = 2000
    results = cudaq.run(compare_superposed, shots_count=shots)
    mismatches = 0
    agreement = Counter()
    for r in results:
        a, b = r[7], r[8]
        nand_xor, cnot_xor = r[5], r[6]
        expected = a ^ b
        agreement[(nand_xor == expected, cnot_xor == expected)] += 1
        if nand_xor != expected or cnot_xor != expected:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH a={int(a)} b={int(b)} nand_xor={int(nand_xor)} "
                      f"cnot_xor={int(cnot_xor)} expected={int(expected)}")

    print(f"shots={shots}  mismatches={mismatches}")
    print(f"(nand_xor correct, cnot_xor correct) counts: {dict(agreement)}")

    if all_ok and mismatches == 0:
        print("\nSUCCESS: NAND itself was never a quantum gate (it's not reversible) -")
        print("but a reversible-NAND primitive (Toffoli + an ancilla preset to |1>)")
        print("wired into the classic 4-NAND XOR circuit reproduces XOR exactly,")
        print("matching a native 2-CNOT construction on every fixed case AND on")
        print("every shot sampled from a genuine superposition over both inputs.")
    else:
        print("\nFAILURE: see mismatches above.")
