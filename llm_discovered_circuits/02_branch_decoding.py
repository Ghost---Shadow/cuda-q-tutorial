"""Uncompute a branching computation WITHOUT a history stack.

From "Discovering state-of-the-art quantum circuits with large language
models" (Gu, Kemeny, Yelin, Lukin - arXiv:2610.22367), Figure 1(c) and
Equation (2).

The problem: a coherent circuit that takes a data-dependent branch at every
round overwrites information. To run it backwards you need to know which
branch each round took. The naive fix is a history stack - one fresh qubit
per round, holding that round's branch bit forever (until the very end of
the uncompute). The stack grows with the number of rounds.

The paper's observation: sometimes the branch bits are already sitting in
the registers you were keeping anyway. Take a register `path` that starts at
0 and, in round j, adds branch_j * 2^j:

    path_{j+1} = path_j + branch_j * 2^j          (paper's Equation 2)

Since path_j < 2^j, bit j of path_{j+1} is EXACTLY branch_j. Nothing else
can have touched it. So during uncomputation the circuit just reads bit j of
`path` to recover the branch decision - no separate history, and the single
scratch qubit that held the branch bit can be cleared and reused every round.

Toy computation used here (3 rounds over a 3-bit data register `data`):
  round j: branch_j = data[j] XOR data[j+1]    (computed into one scratch qubit)
           if branch_j: flip data[j]           (the "overwrite")
After the flip, data[j] XOR data[j+1] is always 0 whichever branch ran - the
branch is genuinely no longer recoverable from `data` alone. We verify:
  1. Without a record, the data update is many-to-one (not reversible).
  2. The naive circuit (history stack) and the decoded circuit (reads the
     path register) both produce the exact right forward state.
  3. Running each one backwards returns to the initial superposition with
     every ancilla back at |0>.
  4. The decoded circuit needs fewer qubits.
"""

import numpy as np

import cudaq

NUM_ROUNDS = 3  # one round per data bit


@cudaq.kernel
def compute_branch(branch: cudaq.qubit, data: cudaq.qview, round_index: int):
    """branch ^= data[j] XOR data[j+1]; this is its own inverse."""
    num_bits = data.size()
    x.ctrl(data[round_index], branch)
    x.ctrl(data[(round_index + 1) % num_bits], branch)


@cudaq.kernel
def decoded_forward(data: cudaq.qview, path: cudaq.qview, branch: cudaq.qubit):
    """Forward pass that keeps NO history stack.

    After flipping data[j] the branch qubit is cleared by reading it back
    out of path[j], so the same single scratch qubit serves every round.
    """
    for round_index in range(data.size()):
        compute_branch(branch, data, round_index)
        x.ctrl(branch, path[round_index])      # path += branch * 2^round_index
        x.ctrl(branch, data[round_index])      # the overwriting update
        x.ctrl(path[round_index], branch)      # DECODE: branch = path bit j -> clears scratch


@cudaq.kernel
def decoded_backward(data: cudaq.qview, path: cudaq.qview, branch: cudaq.qubit):
    """Exact inverse of decoded_forward; recovers each branch from path."""
    for round_index in range(data.size() - 1, -1, -1):
        x.ctrl(path[round_index], branch)      # recover branch from path bit j
        x.ctrl(branch, data[round_index])      # undo the overwriting update
        x.ctrl(branch, path[round_index])      # clear path bit j
        compute_branch(branch, data, round_index)  # data is restored, so recompute -> clear


@cudaq.kernel
def stack_forward(data: cudaq.qview, path: cudaq.qview, history: cudaq.qview):
    """Naive forward pass: each round's branch bit gets its own permanent qubit."""
    for round_index in range(data.size()):
        compute_branch(history[round_index], data, round_index)
        x.ctrl(history[round_index], path[round_index])
        x.ctrl(history[round_index], data[round_index])


@cudaq.kernel
def stack_backward(data: cudaq.qview, path: cudaq.qview, history: cudaq.qview):
    for round_index in range(data.size() - 1, -1, -1):
        x.ctrl(history[round_index], data[round_index])
        x.ctrl(history[round_index], path[round_index])
        compute_branch(history[round_index], data, round_index)


@cudaq.kernel
def uniform_data(data: cudaq.qview):
    for i in range(data.size()):
        h(data[i])


@cudaq.kernel
def decoded_roundtrip(num_bits: int, run_backward: bool):
    data = cudaq.qvector(num_bits)
    path = cudaq.qvector(num_bits)
    branch = cudaq.qubit()
    uniform_data(data)
    decoded_forward(data, path, branch)
    if run_backward:
        decoded_backward(data, path, branch)


@cudaq.kernel
def stack_roundtrip(num_bits: int, run_backward: bool):
    data = cudaq.qvector(num_bits)
    path = cudaq.qvector(num_bits)
    history = cudaq.qvector(num_bits)
    uniform_data(data)
    stack_forward(data, path, history)
    if run_backward:
        stack_backward(data, path, history)


def classical_forward(data_value: int) -> tuple[int, int]:
    """Reference: run the toy computation on one data value.

    Returns (final data value, path value)."""
    bits = [(data_value >> k) & 1 for k in range(NUM_ROUNDS)]
    path_value = 0
    for round_index in range(NUM_ROUNDS):
        branch = bits[round_index] ^ bits[(round_index + 1) % NUM_ROUNDS]
        path_value += branch << round_index
        if branch:
            bits[round_index] ^= 1
    return sum(bit << k for k, bit in enumerate(bits)), path_value


def basis_index(bit_values: list[int]) -> int:
    """CUDA-Q's statevector is little-endian: qubit 0 is the least significant index bit."""
    return sum(bit << k for k, bit in enumerate(bit_values))


def expected_forward_state(total_qubits: int) -> np.ndarray:
    expected = np.zeros(2**total_qubits, dtype=complex)
    for data_value in range(2**NUM_ROUNDS):
        final_data, path_value = classical_forward(data_value)
        bits = [(final_data >> k) & 1 for k in range(NUM_ROUNDS)]
        bits += [(path_value >> k) & 1 for k in range(NUM_ROUNDS)]
        if total_qubits == 2 * NUM_ROUNDS + NUM_ROUNDS:
            # naive circuit: the history stack mirrors the path bits exactly
            bits += [(path_value >> k) & 1 for k in range(NUM_ROUNDS)]
        else:
            bits += [0]  # decoded circuit: scratch qubit is back at |0>
        expected[basis_index(bits)] = 1 / np.sqrt(2**NUM_ROUNDS)
    return expected


def expected_initial_state(total_qubits: int) -> np.ndarray:
    expected = np.zeros(2**total_qubits, dtype=complex)
    for data_value in range(2**NUM_ROUNDS):
        bits = [(data_value >> k) & 1 for k in range(NUM_ROUNDS)]
        bits += [0] * (total_qubits - NUM_ROUNDS)
        expected[basis_index(bits)] = 1 / np.sqrt(2**NUM_ROUNDS)
    return expected


def overlap(expected: np.ndarray, kernel, run_backward: bool) -> float:
    state = np.array(cudaq.get_state(kernel, NUM_ROUNDS, run_backward), dtype=complex)
    return abs(np.vdot(expected, state))


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    print("Toy computation: for each round j, branch = data[j] XOR data[j+1]; "
          "if branch: flip data[j]\n")

    outputs = [classical_forward(value)[0] for value in range(2**NUM_ROUNDS)]
    print(f"1. Without a record, {2**NUM_ROUNDS} inputs map to only "
          f"{len(set(outputs))} distinct final data values -> not reversible.")
    print(f"   final data values: {outputs}")
    assert len(set(outputs)) < 2**NUM_ROUNDS

    print("\n2. With the path register (path += branch * 2^j) it IS reversible:")
    pairs = [classical_forward(value) for value in range(2**NUM_ROUNDS)]
    assert len(set(pairs)) == 2**NUM_ROUNDS
    for value, (final_data, path_value) in enumerate(pairs):
        print(f"   data={value:03b}  ->  data'={final_data:03b}, path={path_value:03b}")

    decoded_qubits = 2 * NUM_ROUNDS + 1
    stack_qubits = 3 * NUM_ROUNDS

    print("\n3. Exact circuit checks on a superposition of all 8 inputs:")
    for name, kernel, qubits in (("history stack", stack_roundtrip, stack_qubits),
                                 ("decoded", decoded_roundtrip, decoded_qubits)):
        forward = overlap(expected_forward_state(qubits), kernel, False)
        restored = overlap(expected_initial_state(qubits), kernel, True)
        print(f"   {name:>13} ({qubits} qubits): forward state overlap = {forward:.6f}, "
              f"after uncompute overlap with initial state = {restored:.6f}")
        assert np.isclose(forward, 1.0) and np.isclose(restored, 1.0)

    print(f"\n4. The decoded circuit needs {decoded_qubits} qubits instead of "
          f"{stack_qubits}: the history stack grows by one qubit per round "
          f"({NUM_ROUNDS} here), while the decoded version reuses one scratch "
          f"qubit forever.")
    print("   Both circuits restore the initial state exactly and leave every "
          "ancilla at |0>.")
