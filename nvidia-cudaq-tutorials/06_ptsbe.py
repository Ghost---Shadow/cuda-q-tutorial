"""
Tutorial 6: PTSBE - Pre-Trajectory Sampling with Batch Execution

Rewritten by hand from NVIDIA's official CUDA-Q "PTSBE" tutorial:
https://nvidia.github.io/cuda-quantum/latest/using/examples/ptsbe.html

Background: simulating a *noisy* circuit means each shot is its own
random "trajectory" - at every noisy gate, the simulator randomly
picks one Kraus operator (one physical error, e.g. "nothing happened"
or "a bit flip happened") and applies it, then measures. Ordinarily,
plain cudaq.sample() draws a brand new random trajectory - and builds a
brand new state vector - for every single shot. If you ask for 100,000
shots, that's 100,000 state vectors, even though many of those
trajectories are actually identical to each other.

PTSBE's idea: pre-generate a *small* set of unique trajectories first
(e.g. 200 of them), simulate each one exactly once, and then divide the
100,000 shots amongst those 200 pre-computed trajectories according to
how likely each one was. Same statistics, far fewer simulations.

Covers:
  - Building a cudaq.NoiseModel with a readout (measurement) bit-flip
    channel
  - Plain noisy cudaq.sample() as the baseline
  - cudaq.ptsbe.sample() with a capped number of trajectories
  - Inspecting the trajectories PTSBE actually chose, via
    return_execution_data
"""

import cudaq
from cudaq import ptsbe


@cudaq.kernel
def noisy_bell_pair():
    """Same Bell-pair circuit as hello_cudaq.py. The noise itself isn't
    written into the kernel - it's layered on separately via a
    NoiseModel, so the same ideal circuit can be run clean or noisy
    without touching this function."""
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


def build_noise_model(bit_flip_probability: float) -> cudaq.NoiseModel:
    """Attaches independent readout (measurement) noise to each qubit.

    A bit-flip channel with probability p means: with probability p,
    the measured bit comes back flipped from what it "should" be;
    otherwise it's reported correctly. This models a noisy detector,
    one of the most common real sources of error on today's hardware.

    Note this is deliberately *not* placed on the H or CNOT gates
    themselves: noise applied to the gates that build the entangled
    state doesn't necessarily leak into the measured 00/11 statistics
    at all (a depolarizing error on the control qubit before the CNOT,
    for instance, still gets faithfully copied onto the target qubit -
    the entangling correlation survives many single-qubit gate errors).
    Readout noise, applied after entanglement is already locked in, is
    the simplest way to actually see 01/10 outcomes leak into this
    circuit's histogram.
    """
    noise = cudaq.NoiseModel()
    channel = cudaq.BitFlipChannel(bit_flip_probability)
    noise.add_channel("mz", [0], channel)
    noise.add_channel("mz", [1], channel)
    return noise


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(42)

    noise_model = build_noise_model(bit_flip_probability=0.1)
    shots = 20000

    print("=== Baseline: plain noisy cudaq.sample() ===")
    # Every one of the 20,000 shots gets its own freshly-drawn noise
    # trajectory and its own state-vector simulation. Expect mostly 00
    # and 11, but with some 01/10 leaking in now, since noise can flip
    # a qubit after entanglement was already established.
    baseline_result = cudaq.sample(noisy_bell_pair,
                                    shots_count=shots,
                                    noise_model=noise_model)
    print(baseline_result)

    print()
    print("=== PTSBE: same noise, capped at 200 unique trajectories ===")
    # Pre-samples only 200 distinct noise trajectories (instead of up
    # to 20,000), simulates each exactly once, then allocates the full
    # 20,000 shots across them proportionally to how likely each
    # trajectory was. The histogram should closely match the baseline
    # above - PTSBE is a performance optimization, not an approximation
    # of the physics - while running dramatically fewer simulations.
    ptsbe_result = ptsbe.sample(
        noisy_bell_pair,
        shots_count=shots,
        noise_model=noise_model,
        max_trajectories=200,
        sampling_strategy=ptsbe.ProbabilisticSamplingStrategy(seed=42),
        shot_allocation=ptsbe.ShotAllocationStrategy(
            ptsbe.ShotAllocationType.PROPORTIONAL),
        return_execution_data=True,
    )
    print(ptsbe_result)

    print()
    print("=== Inspecting the trajectories PTSBE actually used ===")
    execution_data = ptsbe_result.ptsbe_execution_data
    trajectories = execution_data.trajectories
    print(f"Unique trajectories sampled: {len(trajectories)}")
    # Sort by shot count so the most heavily-used (most probable)
    # trajectories are shown first.
    top_trajectories = sorted(trajectories,
                               key=lambda t: t.num_shots,
                               reverse=True)[:5]
    for i, trajectory in enumerate(top_trajectories):
        print(f"  #{i}: probability={trajectory.probability:.4f}  "
              f"shots_allocated={trajectory.num_shots}")
    print("Notice a small number of trajectories account for most of")
    print("the shots - that concentration is exactly what lets PTSBE")
    print("skip simulating the long tail of rare, low-probability")
    print("trajectories over and over.")
