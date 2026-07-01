"""
Tutorial 3: Executing Kernels

Rewritten by hand from NVIDIA's official CUDA-Q "Executing Kernels" tutorial:
https://nvidia.github.io/cuda-quantum/latest/using/examples/executing_kernels.html

CUDA-Q has four different ways to run a kernel, each answering a
different question:

  - cudaq.sample()    "Over many shots, how often does each classical
                       outcome show up?"        -> a bitstring histogram
  - cudaq.run()       "What did each individual shot return?"
                                                 -> a list of per-shot values
  - cudaq.observe()   "What's the expectation value of some observable
                       (e.g. an energy) on this state?"
                                                 -> a single number
  - cudaq.get_state() "What's the full quantum state vector itself?"
                                                 -> complex amplitudes
                       (simulation only - real hardware can't do this,
                       since reading out a full state vector would
                       require infinitely many measurements)
"""

import cudaq
from cudaq import spin


@cudaq.kernel
def coin_flip_for_sample():
    """cudaq.sample() only accepts kernels that return None (void) -
    it works by inspecting the circuit's classical registers directly,
    not by collecting Python return values."""
    q = cudaq.qubit()
    h(q)
    mz(q)


@cudaq.kernel
def coin_flip_for_run() -> bool:
    """cudaq.run() is the opposite: it requires the kernel to return a
    value, since that's exactly what it hands back to you per shot."""
    q = cudaq.qubit()
    h(q)
    return mz(q)


@cudaq.kernel
def bell_state_unmeasured():
    """Same Bell-state circuit as hello_cudaq.py, but with no `mz` call.

    observe() and get_state() need to inspect the state *before* it
    collapses, so kernels used with them must not measure - measuring
    would destroy exactly the information those two calls want to read.
    """
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== sample(): aggregate histogram over many shots ===")
    # Expected: roughly {0: 500, 1: 500} - a fair coin over 1000 flips.
    histogram = cudaq.sample(coin_flip_for_sample, shots_count=1000)
    print(histogram)

    print()
    print("=== run(): the individual per-shot return values ===")
    # Expected: a list of 10 independent True/False values, in the
    # order they were generated - e.g. [True, False, False, True, ...].
    # Unlike sample(), nothing is aggregated into counts.
    per_shot_results = cudaq.run(coin_flip_for_run, shots_count=10)
    print(per_shot_results)

    print()
    print("=== observe(): expectation value of an operator ===")
    # Z0*Z1 is +1 whenever qubits 0 and 1 agree (00 or 11) and -1
    # whenever they disagree (01 or 10). Because the Bell state only
    # ever collapses to 00 or 11, the two qubits *always* agree, so the
    # expectation value should be exactly +1.0 - a direct numerical
    # signature of the perfect correlation discussed in hello_cudaq.py.
    hamiltonian = spin.z(0) * spin.z(1)
    observe_result = cudaq.observe(bell_state_unmeasured, hamiltonian)
    print(f"<Z0 Z1> = {observe_result.expectation()}")

    print()
    print("=== get_state(): the full state vector ===")
    # Expected amplitudes: roughly [0.707, 0, 0, 0.707] for basis states
    # |00>, |01>, |10>, |11> - i.e. (|00> + |11>) / sqrt(2), matching
    # the Bell state derived by hand in hello_cudaq.py's comments.
    state = cudaq.get_state(bell_state_unmeasured)
    print(state)
