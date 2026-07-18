"""
Can you sort qubits while they're in superposition? Not if "sorting" means
inspecting values and branching on them - that needs mz(), and mz() collapses
the state. But a sorting NETWORK doesn't inspect values at all: it's a fixed
sequence of compare-and-swap modules wired in advance (the same idea as a
hardware sorting network, or Batcher's bitonic sort). Each compare-swap here
is built entirely from reversible gates - Toffoli (x.ctrl with 2 controls) to
compute "is x > y?" into an ancilla qubit, and a Fredkin gate (swap.ctrl) to
conditionally swap the two registers - so the whole sort runs as ONE
unitary, no measurement anywhere in the middle.

That means it's legal to feed it a genuine superposition over many input
tuples at once (via Hadamards, not a classical loop over cases) and let the
network sort every branch coherently and in parallel. The catch: a single
run only ever samples ONE branch when you finally measure - you don't get
"the sorted superposition" as a readable object, you get one shot's input
and that same shot's output. So the real claim being tested is narrower and
more honest than "quantum speedup": every single shot's measured input,
independently sampled from the superposition, comes out correctly sorted,
with the comparator's helper ancillas (anc, gt0) verified clean (uncomputed
back to |0>) every time - nothing about the mechanism is measurement-based
except the unavoidable final readout.

Three layers, cleanest first:
  1. odd_even_transposition_pairs() - a classical algorithm that generates
     the comparator wiring for N elements. This is what actually justifies
     "compare (0,1), then (1,2), then (0,1) again" instead of it being a
     hand-picked magic sequence - it's a textbook sorting network, and it's
     correct for any N.
  2. SortNetworkLayout - turns "N elements, 2 bits each" into a qubit index
     plan (where each element's register lives, where each comparator's
     ancillas live) once, in plain Python, so nothing downstream does index
     arithmetic by hand.
  3. The kernels - compare_swap() operates on whole registers (cudaq.qview),
     not individual qubits, and the network itself is a single generic loop
     over the layout's plan rather than one manually-repeated call per
     comparator.
"""

from dataclasses import dataclass, field

import cudaq

ANCILLAS_PER_COMPARATOR = 3  # anc, gt0, gt - see compare_swap()'s docstring


def odd_even_transposition_pairs(n: int) -> list[tuple[int, int]]:
    """A textbook comparator network: n rounds, alternating between
    comparing (0,1)(2,3).. and (1,2)(3,4).. . Provably sorts any n
    elements (each round can advance the eventual position of any element
    still out of place by at least one slot). For n=3 this generates
    exactly [(0,1), (1,2), (0,1)] - three comparators, not an arbitrary
    choice."""
    pairs = []
    for round_idx in range(n):
        start = round_idx % 2
        for i in range(start, n - 1, 2):
            pairs.append((i, i + 1))
    return pairs


@dataclass(frozen=True)
class SortNetworkLayout:
    """Qubit index plan for sorting `num_elements` values, each
    `bits_per_element` bits wide. Three qubit blocks, back to back:
    shadow (snapshot of the input, for verification only), data (gets
    permuted in place into sorted order), and one ancilla triple per
    comparator round. Exposes flat index lists so the kernel can walk the
    network with a plain for-loop instead of repeating hand-written calls."""
    num_elements: int
    bits_per_element: int = 2
    comparator_pairs: list = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "comparator_pairs",
                            odd_even_transposition_pairs(self.num_elements))

    @property
    def data_qubits(self) -> int:
        return self.num_elements * self.bits_per_element

    @property
    def shadow_qubits(self) -> int:
        return self.data_qubits

    @property
    def num_comparators(self) -> int:
        return len(self.comparator_pairs)

    @property
    def total_qubits(self) -> int:
        return (self.shadow_qubits + self.data_qubits
                + self.num_comparators * ANCILLAS_PER_COMPARATOR)

    def shadow_start(self, element_index: int) -> int:
        return element_index * self.bits_per_element

    def element_start(self, element_index: int) -> int:
        return self.shadow_qubits + element_index * self.bits_per_element

    def comparator_anc_start(self, round_index: int) -> int:
        return (self.shadow_qubits + self.data_qubits
                + round_index * ANCILLAS_PER_COMPARATOR)

    def left_starts(self) -> list[int]:
        return [self.element_start(i) for i, _ in self.comparator_pairs]

    def right_starts(self) -> list[int]:
        return [self.element_start(j) for _, j in self.comparator_pairs]

    def anc_starts(self) -> list[int]:
        return [self.comparator_anc_start(k) for k in range(self.num_comparators)]


NUM_ELEMENTS = 3  # values 0..3, 2 bits each
LAYOUT = SortNetworkLayout(num_elements=NUM_ELEMENTS)

# Kernels can only index qubits using plain int/list-of-int globals resolved
# at compile time (no method calls, no dataclass properties) - so the plan
# above gets flattened once, here, into exactly that.
BITS_PER_ELEMENT = LAYOUT.bits_per_element
DATA_QUBITS = LAYOUT.data_qubits
SHADOW_QUBITS = LAYOUT.shadow_qubits
NUM_COMPARATORS = LAYOUT.num_comparators
TOTAL_QUBITS = LAYOUT.total_qubits
DATA_START = [LAYOUT.element_start(i) for i in range(NUM_ELEMENTS)]
SHADOW_START = [LAYOUT.shadow_start(i) for i in range(NUM_ELEMENTS)]
LEFT_START = LAYOUT.left_starts()
RIGHT_START = LAYOUT.right_starts()
ANC_START = LAYOUT.anc_starts()


@cudaq.kernel
def compare_swap(x_reg: cudaq.qview, y_reg: cudaq.qview,
                  anc: cudaq.qubit, gt0: cudaq.qubit, gt: cudaq.qubit):
    """If x > y (2-bit unsigned, x = 2*x_reg[0] + x_reg[1]), swap x_reg and
    y_reg so the smaller value ends up in x_reg. gt = 1 iff x_reg[0] >
    y_reg[0], or (they're equal and x_reg[1] > y_reg[1]) - computed via
    Toffolis with temporary NOT-flips (no negative-control primitive
    needed), then used as the Fredkin gate's control. anc and gt0 are
    uncomputed back to |0> before returning; gt is left holding the "did
    this comparator swap" decision bit."""
    x1, x0 = x_reg[0], x_reg[1]
    y1, y0 = y_reg[0], y_reg[1]

    x.ctrl(x1, anc)
    x.ctrl(y1, anc)  # anc = x1 XOR y1

    x(y1)
    x.ctrl(x1, y1, gt)  # gt = x1 AND (NOT y1)  ->  "x1 > y1"
    x(y1)

    x(anc)  # anc now reads 1 iff x1 == y1

    x(y0)
    x.ctrl(x0, y0, gt0)  # gt0 = x0 AND (NOT y0)  ->  "x0 > y0"
    x(y0)

    x.ctrl(anc, gt0, gt)  # gt ^= (x1==y1) AND (x0>y0); mutually exclusive with the first term, so XOR = OR

    x(y0)
    x.ctrl(x0, y0, gt0)  # uncompute gt0
    x(y0)

    x(anc)

    x.ctrl(y1, anc)  # uncompute anc
    x.ctrl(x1, anc)

    swap.ctrl(gt, x1, y1)
    swap.ctrl(gt, x0, y0)


@cudaq.kernel
def _snapshot_and_sort(q: cudaq.qvector):
    """Copy whatever's currently in the data qubits into the shadow qubits
    (a per-branch classical snapshot, not a clone of an unknown state - see
    module docstring), then walk the comparator network exactly as laid
    out by SortNetworkLayout: one compare_swap per round, addressed by
    register instead of by individual qubit."""
    for i in range(NUM_ELEMENTS):
        ds = DATA_START[i]
        ss = SHADOW_START[i]
        for b in range(BITS_PER_ELEMENT):
            x.ctrl(q[ds + b], q[ss + b])

    for k in range(NUM_COMPARATORS):
        ls = LEFT_START[k]
        rs = RIGHT_START[k]
        a = ANC_START[k]
        x_reg = q[ls:ls + BITS_PER_ELEMENT]
        y_reg = q[rs:rs + BITS_PER_ELEMENT]
        compare_swap(x_reg, y_reg, q[a], q[a + 1], q[a + 2])


@cudaq.kernel
def sort_fixed(input_bits: list[bool]) -> list[bool]:
    """A single definite (non-superposed) input, encoded bit-for-bit:
    input_bits[i]=True sets data qubit i to |1>, False leaves it |0>."""
    q = cudaq.qvector(TOTAL_QUBITS)
    for i in range(DATA_QUBITS):
        if input_bits[i]:
            x(q[SHADOW_QUBITS + i])
    _snapshot_and_sort(q)
    return mz(q)


@cudaq.kernel
def sort_superposed() -> list[bool]:
    """All data qubits Hadamard'd: an equal superposition over every
    possible input tuple, sorted by the SAME unitary network - the
    shadow-copy snapshot happens per-branch, before any comparator runs."""
    q = cudaq.qvector(TOTAL_QUBITS)
    for i in range(DATA_QUBITS):
        h(q[SHADOW_QUBITS + i])
    _snapshot_and_sort(q)
    return mz(q)


def _register_value(bits, start: int) -> int:
    value = 0
    for b in range(LAYOUT.bits_per_element):
        if bits[start + b]:
            value += 1 << (LAYOUT.bits_per_element - 1 - b)
    return value


def decode(bits) -> tuple[tuple[int, ...], tuple[int, ...], list[bool], list[bool]]:
    """Turn a raw measurement bitstring back into (input tuple, output
    tuple, ancilla-garbage bits, per-comparator swap-fired flags) using the
    same SortNetworkLayout the kernel was built from."""
    inputs = tuple(_register_value(bits, LAYOUT.shadow_start(i)) for i in range(NUM_ELEMENTS))
    outputs = tuple(_register_value(bits, LAYOUT.element_start(i)) for i in range(NUM_ELEMENTS))
    ancilla_garbage = []
    swap_flags = []
    for k in range(NUM_COMPARATORS):
        a = LAYOUT.comparator_anc_start(k)
        ancilla_garbage.append(bits[a])
        ancilla_garbage.append(bits[a + 1])
        swap_flags.append(bits[a + 2])
    return inputs, outputs, ancilla_garbage, swap_flags


def _encode_input(values: tuple[int, ...]) -> list[bool]:
    bits = []
    for v in values:
        for b in range(LAYOUT.bits_per_element):
            bits.append(bool(v & (1 << (LAYOUT.bits_per_element - 1 - b))))
    return bits


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)

    print(f"Comparator network for {NUM_ELEMENTS} elements: {LAYOUT.comparator_pairs}")
    print(f"({NUM_COMPARATORS} comparators, {TOTAL_QUBITS} qubits total: "
          f"{LAYOUT.shadow_qubits} shadow + {LAYOUT.data_qubits} data + "
          f"{NUM_COMPARATORS}x{ANCILLAS_PER_COMPARATOR} ancilla)")

    print("\n=== One worked example: fixed input (3, 1, 2) ===")
    fixed_input = (3, 1, 2)
    result = cudaq.run(sort_fixed, _encode_input(fixed_input), shots_count=1)[0]
    inputs, outputs, ancilla_garbage, swap_flags = decode(result)
    print(f"input={inputs}  output={outputs}  expected={tuple(sorted(inputs))}")
    print(f"comparator swap flags {LAYOUT.comparator_pairs}: {swap_flags}")
    print(f"ancilla garbage (should be all 0): {[int(b) for b in ancilla_garbage]}")

    print("\n=== Full superposition: every data qubit Hadamard'd ===")
    num_input_combos = (2 ** LAYOUT.bits_per_element) ** NUM_ELEMENTS
    print(f"Every shot below independently samples one of the {num_input_combos} possible")
    print("input tuples, all processed by the SAME unitary network with no")
    print("measurement in between - only the final mz().")
    shots = 2000
    results = cudaq.run(sort_superposed, shots_count=shots)

    mismatches = 0
    dirty_ancillas = 0
    fire_counts = [0] * NUM_COMPARATORS
    for r in results:
        inputs, outputs, ancilla_garbage, swap_flags = decode(r)
        if list(outputs) != sorted(inputs):
            mismatches += 1
            if mismatches <= 5:
                print(f"  MISMATCH input={inputs} expected={sorted(inputs)} got={outputs}")
        if any(ancilla_garbage):
            dirty_ancillas += 1
        for k, fired in enumerate(swap_flags):
            if fired:
                fire_counts[k] += 1

    print(f"\nshots={shots}")
    print(f"mismatches (output != sorted(input)): {mismatches}")
    print(f"shots with leftover ancilla garbage (should be 0): {dirty_ancillas}")
    fire_rates = ", ".join(f"{pair}={fire_counts[k]/shots:.1%}"
                            for k, pair in enumerate(LAYOUT.comparator_pairs))
    print(f"comparator fire rates: {fire_rates}")
    print("-> The 3rd comparator (0,1) only fires when the first two rounds")
    print("   left element 0 greater than element 1 - exactly like the")
    print("   classical version of this same network.")

    if mismatches == 0 and dirty_ancillas == 0:
        print("\nSUCCESS: every sampled shot came out correctly sorted, and no")
        print("gate ever inspected a qubit's value mid-circuit to decide anything -")
        print("the compare-and-swap decisions were made unitarily, by Toffoli +")
        print("Fredkin gates acting on superposed inputs, and only READ OUT at")
        print("the very end via measurement.")
    else:
        print("\nFAILURE: see mismatches/dirty ancillas above.")
