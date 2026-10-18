"""Measurement-based uncomputation (MBU): clear a temporary AND for one Toffoli.

From "Discovering state-of-the-art quantum circuits with large language
models" (Gu, Kemeny, Yelin, Lukin - arXiv:2610.22367), Methods section,
Equations (11) and (12); the technique itself is Gidney's ("Halving the
cost of quantum addition", arXiv:1709.06648), which the paper's circuit
model builds on.

A temporary AND `t = u AND v` is cheap to COMPUTE (one Toffoli) but the
obvious way to CLEAR it afterwards is to apply the same Toffoli again - a
second Toffoli, doubling the cost of every temporary value. Toffolis are the
expensive non-Clifford resource, so that doubling matters.

MBU replaces the second Toffoli with a measurement:

    |u,v>|0>  --Toffoli-->  |u,v>|uv>
              --(use t)-->  ...
              --measure t in the X basis, outcome r-->  (-1)^(r*u*v) |u,v>|0>
              --CZ(u,v) if r = 1-->                      |u,v>|0>

Measuring t in the X basis releases the ancilla but kicks a sign
(-1)^(r*u*v) back onto the surviving qubits, because t was entangled with
them through t = uv. That sign is a diagonal phase, and CZ is exactly the
(Clifford, cheap) gate that cancels (-1)^(u*v). The measurement outcome r is
uniformly random, so the correction fires half the time on average.

Nothing about this is visible in a single measurement of the output - the
phase error only shows up as wrong INTERFERENCE. So the test here is an
interferometer: build a uniform superposition over (u, v), compute w = u AND v
(the "use" step), clean up t, then UNcompute w and undo the Hadamards. If the
circuit leaves exactly the right state, everything interferes back to |00>
with probability 1. Skipping the CZ correction leaves a leftover sign: half the
shots (r = 0) are still fine, but on the other half the CZ phase makes
P(00) only |<++|CZ|++>|^2 = 1/4, for 0.5 + 0.5 * 0.25 = 0.625 overall.
"""

from collections import Counter

import cudaq


@cudaq.kernel
def and_then_clear(cleanup_style: int) -> list[bool]:
    """cleanup_style: 0 = second Toffoli, 1 = MBU with CZ fix, 2 = MBU, fix forgotten.

    Returns the final (u, v, w) readout of the interferometer.
    """
    u = cudaq.qubit()
    v = cudaq.qubit()
    temp_and = cudaq.qubit()
    w = cudaq.qubit()

    h(u)
    h(v)
    x.ctrl([u, v], temp_and)   # temp_and = u AND v     (Toffoli #1)
    x.ctrl(temp_and, w)        # the "use": copy the AND into w

    if cleanup_style == 0:
        x.ctrl([u, v], temp_and)  # clear temp_and the expensive way (Toffoli #2)
    else:
        outcome = mx(temp_and)    # destructive X-basis measurement frees the ancilla
        reset(temp_and)           # put the (already collapsed) qubit back at |0>
        if cleanup_style == 1:
            if outcome:
                z.ctrl(u, v)      # CZ cancels the (-1)^(u*v) sign kick

    # Interferometer readout: undo the "use" step, then undo the Hadamards.
    x.ctrl([u, v], w)
    h(u)
    h(v)
    return [mz(u), mz(v), mz(w)]


STYLES = [
    (0, "second Toffoli", 2),
    (1, "MBU + CZ correction", 1),
    (2, "MBU, correction forgotten", 1),
]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)

    shots = 4000
    print("Interferometer: expect (u, v, w) = (0, 0, 0) on every shot if the "
          "cleanup leaves the right state.\n")
    print(f"{'cleanup':>28} {'Toffolis':>9} {'P(0,0,0)':>10}")

    results = {}
    for style, name, toffoli_count in STYLES:
        runs = cudaq.run(and_then_clear, style, shots_count=shots)
        counts = Counter(tuple(int(bit) for bit in run) for run in runs)
        success = counts[(0, 0, 0)] / shots
        results[style] = success
        print(f"{name:>28} {toffoli_count:>9} {success:>10.3f}")

    assert results[0] == 1.0, "Toffoli cleanup should be exact"
    assert results[1] == 1.0, "MBU with the CZ correction should be exact"
    # Without the correction: r = 0 half the time (exact), r = 1 half the
    # time (success probability 1/4) -> 0.5 + 0.5 * 0.25 = 0.625.
    assert abs(results[2] - 0.625) < 0.03, "forgetting the fix should give ~0.625"

    print("\nMBU halves the Toffoli cost of the temporary AND (2 -> 1) and is "
          "exactly as correct as the Toffoli cleanup, as long as the CZ "
          "correction is applied.")
    print("Forgetting the correction gives ~0.625 - exactly what the phase kick "
          "predicts: shots with r = 0 are untouched, shots with r = 1 are left "
          "with a CZ phase and only interfere back to (0, 0, 0) 1/4 of the time.")
