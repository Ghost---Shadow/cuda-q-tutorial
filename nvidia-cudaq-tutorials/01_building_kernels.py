"""
Tutorial 1: Building Kernels

Rewritten by hand from NVIDIA's official CUDA-Q "Building Kernels" tutorial:
https://nvidia.github.io/cuda-quantum/latest/using/examples/building_kernels.html

The official example is correct but terse. This version keeps the same
ideas but spells out *why* each step exists, and reuses the same
verbose-gate-wrapper style from hello_cudaq.py in this repo (see that file
for why `h`, `x`, `mz` are special DSL keywords rather than normal Python
names).

Covers:
  - Allocating a register of qubits (a qvector) instead of one at a time
  - Composing kernels: calling one @cudaq.kernel from another
  - Multi-qubit controlled operations, used to build an N-qubit GHZ state
  - Adjoint: the "undo" of a gate, and why gate + its adjoint = identity
"""

import cudaq


@cudaq.kernel
def apply_hadamard(q: cudaq.qubit):
    h(q)


@cudaq.kernel
def apply_cnot(control: cudaq.qubit, target: cudaq.qubit):
    x.ctrl(control, target)


@cudaq.kernel
def ghz_state(n_qubits: int):
    """N-qubit generalization of the 2-qubit Bell state:
    |00...0> + |11...1> (unnormalized).

    Recipe: put qubit 0 into superposition, then chain CNOTs so every
    other qubit copies qubit 0. Since qubit 0 is undetermined, "copying"
    it spreads that same undetermined-ness across all N qubits - they
    all collapse to 0 together, or all collapse to 1 together, never a
    mix of both.
    """
    qubits = cudaq.qvector(n_qubits)
    apply_hadamard(qubits[0])
    for i in range(n_qubits - 1):
        apply_cnot(qubits[i], qubits[i + 1])
    mz(qubits)


@cudaq.kernel
def adjoint_round_trip(q: cudaq.qubit):
    """H, S, S-adjoint, H should leave the qubit exactly as it started.

    S applies a 90-degree phase rotation. `s.adj` applies the opposite
    90-degree rotation. Doing both back to back is the identity
    operation - like turning a dial 90 degrees clockwise then 90 degrees
    counter-clockwise, ending up exactly where you began. The two
    Hadamards bracket this so the phase rotation is actually visible in
    the measurement statistics: H alone would leave the qubit in a
    50/50 superposition regardless of what S does, so we need the
    second H to fold the phase back into a measurable probability.
    """
    h(q)
    s(q)
    s.adj(q)
    h(q)
    mz(q)


@cudaq.kernel
def adjoint_demo():
    """Entry point that allocates the qubit and calls adjoint_round_trip.

    cudaq.sample() and friends need a kernel that owns its own qubit
    allocation - you can't hand sample() a kernel that expects a qubit
    as a parameter, since there'd be no qubit to pass in yet. Composing
    kernels like this (one that allocates, one that operates) is the
    normal pattern for reusing a sub-routine across multiple qubits or
    multiple entry points.
    """
    q = cudaq.qubit()
    adjoint_round_trip(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== GHZ state on 5 qubits ===")
    # Expected: only "00000" and "11111" appear, each ~50% of 1000 shots.
    # Same mechanism as the 2-qubit Bell state, just spread across more
    # qubits - see hello_cudaq.py for the 2-qubit walkthrough.
    ghz_result = cudaq.sample(ghz_state, 5, shots_count=1000)
    print(ghz_result)

    print()
    print("=== Adjoint round trip: H, S, S-adjoint, H ===")
    # Expected: always "0". S and S-adjoint cancel exactly (net-zero
    # phase rotation), and the second H undoes the first - so the qubit
    # deterministically ends up back in |0>, with no randomness left.
    adjoint_result = cudaq.sample(adjoint_demo, shots_count=1000)
    print(adjoint_result)
