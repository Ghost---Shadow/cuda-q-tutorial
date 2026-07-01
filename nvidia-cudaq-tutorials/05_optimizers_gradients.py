"""
Tutorial 5: Optimizers & Gradients

Rewritten by hand from NVIDIA's official CUDA-Q "Optimizers and Gradients"
tutorial:
https://nvidia.github.io/cuda-quantum/latest/examples/python/optimizers_gradients.html

This is a small VQE (Variational Quantum Eigensolver): given a fixed
Hamiltonian and a parameterized circuit ("ansatz"), find the parameter
that makes <H> as small as possible. The result approximates the
Hamiltonian's lowest eigenvalue (its ground-state energy), the same
Hamiltonian used in the official tutorial (a stand-in for a real
molecule's electronic-structure problem at a fixed bond length).

Unlike tutorial 4's Hamiltonian (which we solved by brute-force scanning
theta over the whole circle), this Hamiltonian has enough terms that a
brute-force grid search over one parameter still works, but a real VQE
problem might have dozens of parameters - which is exactly why CUDA-Q
ships proper optimizers instead of asking you to scan by hand.

Covers:
  - A 2-qubit ansatz with one tunable parameter
  - Gradient-based optimization: Adam + finite-difference gradients
  - Gradient-free optimization: SPSA (only needs the cost, not a gradient)
  - Plugging CUDA-Q into a third-party optimizer (SciPy's COBYLA)
"""

import cudaq
from cudaq import spin
from scipy.optimize import minimize


@cudaq.kernel
def ansatz(angle: list[float]):
    """A 2-qubit trial state with a single free parameter.

    Starts both qubits at |0>, flips qubit 0 to |1>, rotates qubit 1 by
    `angle` (spreading it across |0> and |1>), then lets qubit 1
    control a flip back on qubit 0. The net effect: as `angle` sweeps,
    the circuit interpolates between different entangled 2-qubit
    states, and we're searching for the one angle that minimizes the
    Hamiltonian below.
    """
    qubits = cudaq.qvector(2)
    x(qubits[0])
    ry(angle[0], qubits[1])
    x.ctrl(qubits[1], qubits[0])


# The same 2-qubit Hamiltonian used in the official tutorial - a stand-in
# for a real molecule's energy operator at a fixed geometry. The constant
# term (5.907) just shifts every eigenvalue by a fixed amount; the X0*X1
# and Y0*Y1 terms are what actually couple the two qubits together.
hamiltonian = (5.907 - 2.1433 * spin.x(0) * spin.x(1) -
               2.1433 * spin.y(0) * spin.y(1) + 0.21829 * spin.z(0) -
               6.125 * spin.z(1))


def cost(angle_vector):
    """Wraps observe() so every optimizer below can treat this as a
    plain "give me a number for this parameter" black box."""
    return cudaq.observe(ansatz, hamiltonian, angle_vector).expectation()


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    initial_guess = [0.0]

    print("=== Gradient-based: Adam + central-difference gradients ===")
    # Adam needs an actual gradient vector, not just the cost - CUDA-Q
    # doesn't compute that analytically for us here, so we approximate
    # it numerically: nudge the parameter slightly up and down and see
    # how the cost responds (central-difference finite differences).
    adam = cudaq.optimizers.Adam()
    adam.initial_parameters = initial_guess
    adam.max_iterations = 100
    gradient = cudaq.gradients.CentralDifference()

    def cost_and_gradient(angle_vector):
        current_cost = cost(angle_vector)
        grad_vector = gradient.compute(angle_vector, cost, current_cost)
        return current_cost, grad_vector

    adam_energy, adam_params = adam.optimize(dimensions=1,
                                              function=cost_and_gradient)
    print(f"Adam:  energy={adam_energy:.6f}  angle={adam_params[0]:.6f}")

    print()
    print("=== Gradient-free: SPSA ===")
    # SPSA never asks for a gradient at all - it perturbs the parameter
    # randomly and infers a direction to move from just two cost
    # evaluations per step. Useful when a gradient is expensive or
    # unavailable (e.g. on noisy real hardware).
    spsa = cudaq.optimizers.SPSA()
    spsa.initial_parameters = initial_guess
    spsa.max_iterations = 200
    spsa_energy, spsa_params = spsa.optimize(dimensions=1, function=cost)
    print(f"SPSA:  energy={spsa_energy:.6f}  angle={spsa_params[0]:.6f}")

    print()
    print("=== Third-party optimizer: SciPy's COBYLA ===")
    # cudaq.observe() is just a Python function call, so nothing stops
    # you from handing the same cost() function to any optimizer
    # library instead of CUDA-Q's built-in ones.
    scipy_result = minimize(lambda angle_vector: cost(angle_vector),
                             initial_guess,
                             method="COBYLA",
                             options={"maxiter": 100})
    print(f"COBYLA: energy={scipy_result.fun:.6f}  angle={scipy_result.x[0]:.6f}")

    print()
    print("All three should land on the same minimum energy (up to")
    print("solver tolerance), since they're all searching the same")
    print("one-dimensional landscape - just with different strategies.")
