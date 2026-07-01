"""
Tutorial 2: Measuring Kernels

Rewritten by hand from NVIDIA's official CUDA-Q "Measuring Kernels" tutorial:
https://nvidia.github.io/cuda-quantum/latest/using/examples/measuring_kernels.html

Covers:
  - Mid-circuit measurement: measuring a qubit *before* the kernel ends,
    and capturing the classical result in a variable
  - Using that classical result in an `if` statement to control later
    gates (a "classically-controlled" or "feed-forward" circuit)
  - Why this is a genuinely different mechanism from entanglement, even
    though it can produce a similar-looking output histogram
"""

from collections import Counter

import cudaq


@cudaq.kernel
def entangle_via_measurement_feedforward() -> list[bool]:
    """Puts qubit 0 into superposition, measures it *mid-circuit*, and
    uses that classical bit to decide whether to flip qubit 1.

    This is deliberately built to mimic the Bell-state histogram from
    hello_cudaq.py - {00, 11} roughly 50/50 - but the mechanism is
    completely different:

    - In hello_cudaq.py's Bell state, both qubits stay in a joint
      superposition until measurement; entanglement is what makes them
      agree.
    - Here, qubit 0 is collapsed to a definite classical bit first, and
      qubit 1 is then deterministically set to match it via an
      ordinary classical `if`. There's no entanglement left by the time
      qubit 1 gets its value - it's a classical copy operation.

    Same statistics, different physics. This is exactly the kind of
    "hidden variable" mechanism Bell's theorem experiments rule out for
    real entangled qubits - see the discussion in hello_cudaq.py.

    Kernels that branch on a measurement result must explicitly return
    the bits they measured - this version of CUDA-Q no longer allows
    cudaq.sample() on such kernels (it needs cudaq.run() instead, since
    branching kernels can't be represented as one static circuit the
    way sample() expects; see the histogram reconstruction below).
    """
    qubits = cudaq.qvector(2)
    h(qubits[0])
    bit0 = mz(qubits[0])
    if bit0:
        x(qubits[1])
    bit1 = mz(qubits[1])
    return [bit0, bit1]


@cudaq.kernel
def measurement_reset_demo() -> bool:
    """Measures a qubit mid-circuit, and returns that classical bit
    directly from the kernel for use in ordinary Python afterwards.

    Demonstrates that a mid-circuit measurement handle isn't just for
    `if` statements inside the kernel - it's a normal boolean value you
    can pass around, combine, or return like any other.
    """
    q = cudaq.qubit()
    h(q)
    return mz(q)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    print("=== Measurement + classical feed-forward ===")
    # cudaq.run() gives back the two measured bits for every individual
    # shot (a branching kernel can't go through cudaq.sample() - see
    # the kernel's docstring above). We rebuild a sample()-style
    # histogram by hand with Counter, purely for comparison.
    # Expected: only "00" and "11" appear, ~50/50 - visually identical
    # to the Bell-state histogram, despite having no entanglement left
    # at the point qubit 1 is set.
    per_shot_bits = cudaq.run(entangle_via_measurement_feedforward, shots_count=1000)
    histogram = Counter(
        "".join("1" if bit else "0" for bit in bits) for bits in per_shot_bits)
    print(dict(histogram))

    print()
    print("=== Returning a mid-circuit measurement from the kernel ===")
    # cudaq.run() executes the kernel `shots_count` times and gives back
    # the individual per-shot return values, not a histogram - useful
    # when you need the raw sequence of outcomes rather than aggregate
    # counts. Expect roughly half True, half False, in no particular
    # order, since each shot is an independent coin flip.
    individual_outcomes = cudaq.run(measurement_reset_demo, shots_count=10)
    print(individual_outcomes)
