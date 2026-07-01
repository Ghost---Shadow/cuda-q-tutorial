"""
Tutorial 4: Constructing Operators

Rewritten by hand from NVIDIA's official CUDA-Q "Constructing Operators"
tutorial:
https://nvidia.github.io/cuda-quantum/latest/using/examples/operators.html

Covers:
  - Building spin operators (Pauli Z, X, ...) with cudaq.spin
  - Combining them with +, -, * into a multi-term Hamiltonian, the same
    way you'd write physics on paper (e.g. H = Z0 Z1 + 0.5 X0)
  - Feeding a parameter into a kernel and scanning it to minimize the
    expectation value of that Hamiltonian - the core idea behind
    variational algorithms like VQE, without pulling in a full optimizer
    library. This is "the world's smallest VQE."
"""

import math

import cudaq
from cudaq import spin


@cudaq.kernel
def one_qubit_ansatz(theta: float):
    """A single-qubit trial state parameterized by one angle.

    ry(theta) rotates the qubit from |0> towards |1> by angle theta:
    the state becomes cos(theta/2)|0> + sin(theta/2)|1>. Sweeping theta
    from 0 to 2*pi visits every point on that qubit's Bloch-sphere
    "meridian" - it's the standard one-parameter trial state used to
    demonstrate variational algorithms.
    """
    q = cudaq.qubit()
    ry(theta, q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    # A toy two-term Hamiltonian: H = Z0 + X0.
    # For a single qubit, <Z> = cos(theta) and <X> = sin(theta) under
    # the ansatz above, so <H> = cos(theta) + sin(theta). Its true
    # minimum eigenvalue is -sqrt(2) (both Z and X have eigenvalues
    # +-1, and this particular combination's minimum is -sqrt(1^2+1^2)).
    # We'll rediscover that minimum numerically below, without ever
    # doing the eigenvalue math ourselves - just by scanning theta and
    # asking the simulator for <H> at each point.
    hamiltonian = spin.z(0) + spin.x(0)
    print(f"Hamiltonian: {hamiltonian}")

    print()
    print("=== Scanning theta to minimize <H> ===")
    best_theta = None
    best_energy = math.inf
    for step in range(24):
        theta = 2 * math.pi * step / 24
        energy = cudaq.observe(one_qubit_ansatz, hamiltonian, theta).expectation()
        print(f"theta={theta:.3f}  <H>={energy:.4f}")
        if energy < best_energy:
            best_energy = energy
            best_theta = theta

    print()
    print(f"Lowest <H> found: {best_energy:.4f} at theta={best_theta:.3f}")
    print(f"Analytic minimum: {-math.sqrt(2):.4f}")
    # Expected: best_energy should land close to -1.4142 (-sqrt(2)),
    # the exact analytic minimum - the coarse 24-point scan won't hit
    # it exactly, but should get within the scan's step size.
