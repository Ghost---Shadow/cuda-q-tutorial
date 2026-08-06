"""
The Collatz conjecture: starting from any positive integer n, repeatedly
apply n -> n/2 if n is even, n -> 3n+1 if n is odd - conjectured to always
reach 1 eventually. "Put the two branches in superposition" means: don't
pick a branch by looking at n and deciding classically. Instead, keep n in
superposition and let EACH branch's own parity qubit route it through the
correct half of the step - both branches genuinely exist in the circuit at
once, coherently, and only collapse to one concrete trajectory when
measured at the very end.

The obstacle: the Collatz step itself isn't reversible as a bare function.
Both n=8 (even -> 4) and n=1 (odd -> 3*1+1=4) map to 4 - two different
inputs, one output - so it can't be a unitary permutation on its own, same
problem as classical NAND. The fix is the same one used throughout this
repo (reversible-NAND, the sorting network's comparator, the quantum hash
map): embed the step as a lookup-table oracle, |current>|0> ->
|current>|step(current)>, writing the result into a FRESH register instead
of overwriting current in place. current stays around afterward as a
witness of what led to that result - exactly the "shadow" qubit trick from
quantum_sorting_network.py, except here every intermediate register in the
chain doubles as its own shadow for the step after it.

This means the qubit count grows with the number of steps kept around: 1
register per step, each `VALUE_WIDTH` qubits wide (each register's lookup
oracle is built for the SPECIFIC, classically-precomputed set of reachable
values at that step - not a generic arbitrary-precision adder/multiplier,
which would need much more machinery to build reversibly). This demo tracks
4 starting values for 4 steps: 2 (selector) + 5 registers x 5-bit width =
27 qubits, right at the ~1GB single-precision statevector line - the
practical ceiling for exact simulation here, same kind of qubit-budget
tradeoff as the sorting network and the electron monitor.

Verification is exact per shot: every one of the 5 registers (the 4
starting values' shared circuit, entangled together via a shared selector)
gets measured, decoded back to integers, and checked against the classical
Collatz trajectory for whichever starting value that shot landed on - not
just the final answer, the WHOLE path, every step, every shot.
"""

import cudaq

SELECTOR_WIDTH = 2       # 2 qubits -> exactly 4 equally-likely starting values
VALUE_WIDTH = 5          # covers every value reached (max 16) with headroom
NUM_ENTRIES = 4          # matches len(START_VALUES) - no branch merges among them within NUM_STEPS
NUM_STEPS = 4
START_VALUES = [2, 3, 5, 6]
TOTAL_QUBITS = SELECTOR_WIDTH + (NUM_STEPS + 1) * VALUE_WIDTH  # 27


def collatz_step(n: int) -> int:
    """Collatz's rule, with 1 as an explicit absorbing fixed point (so a
    trajectory that arrives early just stays put instead of cycling
    1->4->2->1 forever)."""
    if n == 1:
        return 1
    return n // 2 if n % 2 == 0 else 3 * n + 1


def bits_of(value: int, width: int) -> list[bool]:
    return [bool((value >> b) & 1) for b in range(width)]


def flatten_table(in_vals: list[int], out_vals: list[int], in_width: int, out_width: int):
    in_bits, out_bits = [], []
    for v in in_vals:
        in_bits.extend(bits_of(v, in_width))
    for v in out_vals:
        out_bits.extend(bits_of(v, out_width))
    return in_bits, out_bits


@cudaq.kernel
def apply_lookup_selector(selector: cudaq.qview, target: cudaq.qview,
                           in_bits: list[bool], out_bits: list[bool]):
    """Writes target = table[selector] for NUM_ENTRIES (selector_pattern ->
    value_pattern) pairs: temporarily X-flip whichever selector qubits
    should read 0 for this entry, multi-controlled-X each target bit that
    should become 1, then flip back - the same reversible-lookup pattern as
    quantum_hash_set.py, just with runtime-supplied table data instead of a
    hardcoded pattern."""
    for k in range(NUM_ENTRIES):
        for b in range(SELECTOR_WIDTH):
            if not in_bits[k * SELECTOR_WIDTH + b]:
                x(selector[b])
        for b2 in range(VALUE_WIDTH):
            if out_bits[k * VALUE_WIDTH + b2]:
                x.ctrl(selector[0], selector[1], target[b2])
        for b in range(SELECTOR_WIDTH):
            if not in_bits[k * SELECTOR_WIDTH + b]:
                x(selector[b])


@cudaq.kernel
def apply_lookup_step(current: cudaq.qview, next_: cudaq.qview,
                       in_bits: list[bool], out_bits: list[bool]):
    """Writes next_ = collatz_step(current) for whichever NUM_ENTRIES
    values are actually reachable at this step (precomputed classically -
    this is NOT a generic reversible arithmetic circuit, it only needs to
    be correct on the values that can genuinely appear here)."""
    for k in range(NUM_ENTRIES):
        for b in range(VALUE_WIDTH):
            if not in_bits[k * VALUE_WIDTH + b]:
                x(current[b])
        for b2 in range(VALUE_WIDTH):
            if out_bits[k * VALUE_WIDTH + b2]:
                x.ctrl(current[0], current[1], current[2], current[3], current[4], next_[b2])
        for b in range(VALUE_WIDTH):
            if not in_bits[k * VALUE_WIDTH + b]:
                x(current[b])


@cudaq.kernel
def collatz_superposition(sel_in: list[bool], sel_out: list[bool],
                           s1_in: list[bool], s1_out: list[bool],
                           s2_in: list[bool], s2_out: list[bool],
                           s3_in: list[bool], s3_out: list[bool],
                           s4_in: list[bool], s4_out: list[bool]) -> list[bool]:
    q = cudaq.qvector(TOTAL_QUBITS)
    selector = q[0:SELECTOR_WIDTH]
    h(selector[0])
    h(selector[1])  # equal superposition over all 4 starting values at once

    r0 = q[SELECTOR_WIDTH:SELECTOR_WIDTH + VALUE_WIDTH]
    r1 = q[SELECTOR_WIDTH + VALUE_WIDTH:SELECTOR_WIDTH + 2 * VALUE_WIDTH]
    r2 = q[SELECTOR_WIDTH + 2 * VALUE_WIDTH:SELECTOR_WIDTH + 3 * VALUE_WIDTH]
    r3 = q[SELECTOR_WIDTH + 3 * VALUE_WIDTH:SELECTOR_WIDTH + 4 * VALUE_WIDTH]
    r4 = q[SELECTOR_WIDTH + 4 * VALUE_WIDTH:SELECTOR_WIDTH + 5 * VALUE_WIDTH]

    apply_lookup_selector(selector, r0, sel_in, sel_out)
    # Each step's oracle is a DIFFERENT unitary (different table data) -
    # every branch takes its own correct half of "n/2 or 3n+1" coherently,
    # entangled with its own history, with no measurement in between.
    apply_lookup_step(r0, r1, s1_in, s1_out)
    apply_lookup_step(r1, r2, s2_in, s2_out)
    apply_lookup_step(r2, r3, s3_in, s3_out)
    apply_lookup_step(r3, r4, s4_in, s4_out)

    return mz(q)


def val_of(bits) -> int:
    return sum((1 << b) for b, bit in enumerate(bits) if bit)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(3)

    # Precompute every step's reachable-value lookup table classically.
    trajectories = {n: [n] for n in START_VALUES}
    current_vals = list(START_VALUES)
    step_tables = []
    for _ in range(NUM_STEPS):
        next_vals = [collatz_step(v) for v in current_vals]
        step_tables.append((list(current_vals), list(next_vals)))
        for i, n in enumerate(START_VALUES):
            trajectories[n].append(next_vals[i])
        current_vals = next_vals

    print("Classical ground truth:")
    for n in START_VALUES:
        print(f"  n={n}: {trajectories[n]}")

    selector_in_bits, selector_out_bits = flatten_table(
        [0, 1, 2, 3], START_VALUES, SELECTOR_WIDTH, VALUE_WIDTH)
    step_bit_tables = [
        flatten_table(in_vals, out_vals, VALUE_WIDTH, VALUE_WIDTH)
        for in_vals, out_vals in step_tables
    ]

    args = [selector_in_bits, selector_out_bits]
    for in_bits, out_bits in step_bit_tables:
        args.extend([in_bits, out_bits])

    shots = 2000
    print(f"\nRunning {TOTAL_QUBITS}-qubit circuit, {shots} shots...")
    results = cudaq.run(collatz_superposition, *args, shots_count=shots)

    mismatches = 0
    from collections import Counter
    observed_starts = Counter()
    for r in results:
        offset = SELECTOR_WIDTH
        regs = []
        for _ in range(NUM_STEPS + 1):
            regs.append(val_of(r[offset:offset + VALUE_WIDTH]))
            offset += VALUE_WIDTH
        n0 = regs[0]
        observed_starts[n0] += 1
        if regs != trajectories[n0]:
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH: measured={regs}  expected={trajectories[n0]}")

    print(f"\nshots={shots}  mismatches={mismatches}")
    print(f"starting-value sample distribution: {dict(observed_starts)}  "
          f"(should be roughly even across {START_VALUES})")

    if mismatches == 0:
        print("\nSUCCESS: every shot's full 5-register trajectory - not just the final")
        print("value, the WHOLE path - matches that shot's classical Collatz walk")
        print("exactly. All 4 starting values were explored in one coherent")
        print("superposition; no branch was ever picked by inspecting n mid-circuit,")
        print("only by measuring the whole system once at the very end.")
    else:
        print("\nFAILURE: see mismatches above.")
