"""
Tutorial 7: Quantum Teleportation

Rewritten by hand from NVIDIA's official CUDA-Q "Quantum Teleportation"
tutorial:
https://nvidia.github.io/cuda-quantum/latest/applications/python/quantum_teleportation.html

The goal: move an unknown quantum state from qubit 0 ("the message") onto
qubit 2, using only an entangled pair and two classical bits - without
ever directly measuring or copying qubit 0's state (which is impossible
anyway; the "no-cloning theorem" forbids copying an unknown quantum
state). Nothing here transmits information faster than light: the two
classical bits still have to be sent to wherever qubit 2 is, by an
ordinary classical channel, before the protocol is complete.

Qubit roles:
  - qubits[0]: the "message" qubit - the unknown state we want to move
  - qubits[1], qubits[2]: a pre-shared Bell pair (see hello_cudaq.py) -
    qubits[1] stays with the sender, qubits[2] is wherever the state
    should end up

Protocol:
  1. Prepare qubits[1] and qubits[2] as an entangled Bell pair.
  2. Entangle the message qubit with the sender's half of the pair
     (CNOT then Hadamard) - this is the step that mixes the message's
     information into a form only recoverable via measurement.
  3. Measure qubits[0] and qubits[1], producing two classical bits.
  4. Send those two bits (classically) to whoever holds qubits[2], and
     have them apply a Z gate if the first bit was 1, and an X gate if
     the second bit was 1.
  5. qubits[2] now holds the original message state exactly - even
     though it started this protocol as one half of a Bell pair with
     no relation at all to the message.
"""

import math

import cudaq


@cudaq.kernel
def teleport_and_measure_z(message_angle: float) -> bool:
    """Teleports a message qubit prepared by ry(message_angle), then
    measures the destination qubit directly (Z basis).

    ry(message_angle) prepares qubits[0] as cos(angle/2)|0> +
    sin(angle/2)|1>, so its Z-expectation is cos(message_angle) - a
    single tunable "unknown state" to teleport and check.
    """
    qubits = cudaq.qvector(3)
    ry(message_angle, qubits[0])

    # Step 1: entangle qubits[1] and qubits[2] into a Bell pair.
    h(qubits[1])
    x.ctrl(qubits[1], qubits[2])

    # Step 2: entangle the message qubit with the sender's half of the
    # pair. This doesn't move any information yet - it just correlates
    # the message with the measurement basis we're about to use.
    x.ctrl(qubits[0], qubits[1])
    h(qubits[0])

    # Step 3: measure both of the sender's qubits.
    sender_bit_0 = mz(qubits[0])
    sender_bit_1 = mz(qubits[1])

    # Step 4: classically-controlled corrections on the receiver's
    # qubit. Depending on which of the 4 possible outcomes the sender
    # measured, qubits[2] is currently the message state up to a known,
    # fixable Pauli error - these corrections undo exactly that error.
    if sender_bit_0:
        z(qubits[2])
    if sender_bit_1:
        x(qubits[2])

    # Step 5: qubits[2] now holds the original message state. Measure
    # it directly to check the Z-basis statistics match the original.
    return mz(qubits[2])


@cudaq.kernel
def teleport_and_measure_x(message_angle: float) -> bool:
    """Identical to teleport_and_measure_z, but measures the
    destination qubit in the X basis instead (via H then mz) - lets us
    check that teleportation preserved the *full* quantum state, not
    just its Z-basis statistics."""
    qubits = cudaq.qvector(3)
    ry(message_angle, qubits[0])

    h(qubits[1])
    x.ctrl(qubits[1], qubits[2])

    x.ctrl(qubits[0], qubits[1])
    h(qubits[0])

    sender_bit_0 = mz(qubits[0])
    sender_bit_1 = mz(qubits[1])

    if sender_bit_0:
        z(qubits[2])
    if sender_bit_1:
        x(qubits[2])

    h(qubits[2])
    return mz(qubits[2])


def bit_expectation(bits: list) -> float:
    """Converts a list of measured booleans into a +-1 expectation
    value: False (bit 0) counts as +1, True (bit 1) counts as -1 -
    the usual convention for a Pauli-Z-style measurement."""
    return sum(-1 if bit else 1 for bit in bits) / len(bits)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)
    shots = 4000

    # Note: cudaq.observe() refuses kernels containing mz() calls (it
    # needs an unmeasured state to compute an expectation value
    # analytically) - teleportation *requires* mid-circuit measurement,
    # so we instead verify correctness the way a real experiment would:
    # by measuring qubits[2] many times, in two different bases, and
    # comparing the resulting statistics to the message qubit's known
    # analytic expectation values.
    print("angle    <Z> measured  <Z> expected   <X> measured  <X> expected")
    for message_angle in [0.0, math.pi / 3, math.pi / 2, 2.1]:
        z_bits = cudaq.run(teleport_and_measure_z, message_angle, shots_count=shots)
        x_bits = cudaq.run(teleport_and_measure_x, message_angle, shots_count=shots)
        z_measured = bit_expectation(z_bits)
        x_measured = bit_expectation(x_bits)
        z_expected = math.cos(message_angle)
        x_expected = math.sin(message_angle)
        print(f"{message_angle:5.3f}    {z_measured:11.4f}  {z_expected:12.4f}"
              f"   {x_measured:11.4f}  {x_expected:12.4f}")

    print()
    print("Expected: measured columns should land close to their")
    print("corresponding expected columns (within shot noise, ~1000")
    print("shots gives roughly +-0.03 spread) for every angle - proof")
    print("that qubits[2] ends up in the same state qubits[0] started")
    print("in, for any angle, not just the ones we happened to test.")
