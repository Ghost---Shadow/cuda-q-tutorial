"""Conditional routing: swap the DATA, not the arithmetic.

From "Discovering state-of-the-art quantum circuits with large language
models" (Gu, Kemeny, Yelin, Lukin - arXiv:2610.22367), Figure 1(b) and
Equation (1). The task: given a selector qubit `selector` and three
registers `x`, `y`, `z`,

    x  ->  x + y   if selector = 0
    x  ->  x + z   if selector = 1

leaving y, z and the selector untouched.

A classical program would branch: "if selector, add z, else add y". A
quantum circuit can't branch - the selector may be in superposition - so
the naive translation is TWO controlled adders, one per branch. Both are
always paid for, even though only one of them applies to any given basis
state.

The paper's trick: the two branches are the same arithmetic on different
operands. So use a cheap controlled-swap to route the right operand into a
shared slot, run ONE unconditional adder, and swap back:

    controlled-swap(selector; y, z)   <- cheap (n Fredkin gates)
    x += y                            <- the expensive part, paid once
    controlled-swap(selector; y, z)   <- undo the routing

This is the same idea as bucket-brigade QRAM: where a classical program
routes *execution* to the chosen operation, the coherent circuit routes the
chosen *data* through a shared operation.

The adder here is a plain ripple-carry adder mod 2^n built from Toffolis;
the paper's multiply-add (x += c*y) is just repeated controlled adders, so
the same argument applies one level up.

Both circuits are verified on a full superposition over every possible
(x, y, z, selector) at once, via an exact `get_state()` comparison against
the classical answer - including that every ancilla returns to |0>.
"""

import numpy as np

import cudaq

WIDTH = 3  # bits per register; arithmetic is mod 2**WIDTH
MODULUS = 2**WIDTH


@cudaq.kernel
def add_into(target: cudaq.qview, addend: cudaq.qview, carries: cudaq.qview):
    """target += addend (mod 2^n) with a ripple-carry adder.

    carries[i] temporarily holds the carry INTO bit i+1. Carries are computed
    low-to-high, the top bit is summed directly, then carries are uncomputed
    high-to-low (each one needs the target bit below it to still be
    un-summed), so every carry qubit ends back at |0>.
    """
    width = target.size()
    for i in range(width - 1):
        x.ctrl([target[i], addend[i]], carries[i])
        if i > 0:
            x.ctrl([carries[i - 1], target[i]], carries[i])
            x.ctrl([carries[i - 1], addend[i]], carries[i])

    x.ctrl(addend[width - 1], target[width - 1])
    x.ctrl(carries[width - 2], target[width - 1])

    for i in range(width - 2, -1, -1):
        if i > 0:
            x.ctrl([carries[i - 1], addend[i]], carries[i])
            x.ctrl([carries[i - 1], target[i]], carries[i])
        x.ctrl([target[i], addend[i]], carries[i])
        x.ctrl(addend[i], target[i])
        if i > 0:
            x.ctrl(carries[i - 1], target[i])


@cudaq.kernel
def controlled_add(control: cudaq.qubit, target: cudaq.qview, addend: cudaq.qview,
                   gated_addend: cudaq.qview, carries: cudaq.qview):
    """target += addend, but only if control = 1.

    Copy (control AND addend) into a scratch register, add that, then clear
    the scratch register. Costs an extra 2n Toffolis on top of the adder.
    """
    for i in range(target.size()):
        x.ctrl([control, addend[i]], gated_addend[i])
    add_into(target, gated_addend, carries)
    for i in range(target.size()):
        x.ctrl([control, addend[i]], gated_addend[i])


@cudaq.kernel
def prepare_all_inputs(selector: cudaq.qubit, x_reg: cudaq.qview,
                       y_reg: cudaq.qview, z_reg: cudaq.qview):
    h(selector)
    for i in range(x_reg.size()):
        h(x_reg[i])
        h(y_reg[i])
        h(z_reg[i])


@cudaq.kernel
def naive_branch_circuit(width: int):
    """Two controlled adders: one for each branch."""
    x_reg = cudaq.qvector(width)
    y_reg = cudaq.qvector(width)
    z_reg = cudaq.qvector(width)
    selector = cudaq.qubit()
    carries = cudaq.qvector(width - 1)
    gated_addend = cudaq.qvector(width)
    prepare_all_inputs(selector, x_reg, y_reg, z_reg)

    x(selector)  # run the selector = 0 branch: x += y
    controlled_add(selector, x_reg, y_reg, gated_addend, carries)
    x(selector)
    controlled_add(selector, x_reg, z_reg, gated_addend, carries)  # selector = 1: x += z


@cudaq.kernel
def routed_branch_circuit(width: int):
    """Route the right operand into y's slot, add ONCE, route back."""
    x_reg = cudaq.qvector(width)
    y_reg = cudaq.qvector(width)
    z_reg = cudaq.qvector(width)
    selector = cudaq.qubit()
    carries = cudaq.qvector(width - 1)
    prepare_all_inputs(selector, x_reg, y_reg, z_reg)

    for i in range(width):
        swap.ctrl(selector, y_reg[i], z_reg[i])
    add_into(x_reg, y_reg, carries)
    for i in range(width):
        swap.ctrl(selector, y_reg[i], z_reg[i])


def adder_toffoli_count(width: int) -> int:
    """Toffolis in add_into: carry i=0 costs 1, each later carry costs 3,
    and every carry is computed once and uncomputed once."""
    return 2 * (1 + 3 * (width - 2))


def naive_toffoli_count(width: int) -> int:
    return 2 * (2 * width + adder_toffoli_count(width))


def routed_toffoli_count(width: int) -> int:
    fredkin_gates = 2 * width  # swap in, swap out (one Fredkin ~ one Toffoli)
    return fredkin_gates + adder_toffoli_count(width)


def expected_state(total_qubits: int) -> np.ndarray:
    """Classical answer: a uniform superposition over every input, each one
    mapped to (x + chosen operand mod 2^n, y, z, selector), ancillas |0>.

    CUDA-Q's statevector is little-endian: qubit 0 is the LEAST significant bit of the amplitude index.
    """
    expected = np.zeros(2**total_qubits, dtype=complex)
    num_inputs = MODULUS**3 * 2
    for x_value in range(MODULUS):
        for y_value in range(MODULUS):
            for z_value in range(MODULUS):
                for selector in range(2):
                    chosen = z_value if selector else y_value
                    out_x = (x_value + chosen) % MODULUS
                    bits = []
                    for value in (out_x, y_value, z_value):
                        bits += [(value >> k) & 1 for k in range(WIDTH)]
                    bits.append(selector)
                    bits += [0] * (total_qubits - len(bits))
                    index = sum(bit << k for k, bit in enumerate(bits))
                    expected[index] = 1 / np.sqrt(num_inputs)
    return expected


def check(name: str, kernel, total_qubits: int, toffolis: int) -> None:
    state = np.array(cudaq.get_state(kernel, WIDTH), dtype=complex)
    overlap = abs(np.vdot(expected_state(total_qubits), state))
    print(f"{name:>8}: {total_qubits} qubits, {toffolis} Toffolis, "
          f"overlap with classical answer = {overlap:.6f}")
    assert np.isclose(overlap, 1.0), f"{name} circuit is wrong"


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    print(f"x += (y if selector == 0 else z), {WIDTH}-bit registers, "
          f"checked on all {MODULUS**3 * 2} inputs in superposition\n")

    naive_qubits = 3 * WIDTH + 1 + (WIDTH - 1) + WIDTH
    routed_qubits = 3 * WIDTH + 1 + (WIDTH - 1)
    check("naive", naive_branch_circuit, naive_qubits, naive_toffoli_count(WIDTH))
    check("routed", routed_branch_circuit, routed_qubits, routed_toffoli_count(WIDTH))

    naive_cost = naive_toffoli_count(WIDTH)
    routed_cost = routed_toffoli_count(WIDTH)
    print(f"\nrouting saves {naive_cost - routed_cost} Toffolis "
          f"({naive_cost} -> {routed_cost}, {naive_cost / routed_cost:.1f}x) "
          f"and {naive_qubits - routed_qubits} qubits (no scratch register needed).")
    print("\nThe ratio is exactly 2x at every width, matching the paper's "
          "T_routed = T_naive / 2:")
    for width in (3, 8, 32, 256):
        ratio = naive_toffoli_count(width) / routed_toffoli_count(width)
        print(f"  width {width:>3}: naive {naive_toffoli_count(width):>5}, "
              f"routed {routed_toffoli_count(width):>5}, ratio {ratio:.2f}x")
