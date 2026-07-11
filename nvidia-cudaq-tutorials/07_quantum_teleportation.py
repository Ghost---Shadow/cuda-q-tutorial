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

Code structure is deliberately "nothing up my sleeve": alice_room() is
the only function that ever sees message_angle, and it never receives
qubits[2] - Alice physically cannot act on Bob's qubit. bob_room() is
the only function that touches qubits[2], and its signature has no
message_angle parameter - it only receives the two classical bits
sender_bit_0/sender_bit_1, exactly as the real protocol requires. The
"unknown state" is moved entirely through the shared qubit plus those
two bits, never through a shared Python variable.

What physically moves, made explicit: receiver_qubit (qubits[2]) never
travels anywhere during the protocol - it's already sitting at Bob's
location from the moment prepare_shared_bell_pair() distributes the
Bell pair, and it stays there the whole time. The *only* thing that
crosses from Alice to Bob mid-protocol is 2 classical bits, carried by
classical_channel_to_bob() below - no qubit transport step exists in
this protocol at all. That's the entire point of teleportation, and
the reason it's the mirror image of superdense_coding.py (see that
file's send_qubit_to_bob(), where a qubit *is* the thing physically
carried, and 2 classical bits are what Alice picks locally instead).
"""

import math

import cudaq


@cudaq.kernel
def prepare_shared_bell_pair(sender_qubit: cudaq.qubit, receiver_qubit: cudaq.qubit):
    """The trusted setup phase, run before Alice or Bob do anything: a
    Bell pair, split so Alice keeps one half (sender_qubit) and Bob
    keeps the other (receiver_qubit)."""
    h(sender_qubit)
    x.ctrl(sender_qubit, receiver_qubit)


@cudaq.kernel
def alice_room(message_qubit: cudaq.qubit, sender_qubit: cudaq.qubit,
                message_angle: float) -> list[bool]:
    """Alice's entire world: the message qubit and her half of the Bell
    pair. She never receives receiver_qubit as a parameter, so this
    function is structurally incapable of acting on Bob's qubit -
    everything she sends him has to go through the two returned bits.

    ry(message_angle) prepares the message qubit as
    cos(angle/2)|0> + sin(angle/2)|1>, so its Z-expectation is
    cos(message_angle) - a single tunable "unknown state" to teleport
    and check.
    """
    ry(message_angle, message_qubit)

    # Entangle the message qubit with the sender's half of the pair.
    # This doesn't move any information yet - it just correlates the
    # message with the measurement basis we're about to use.
    x.ctrl(message_qubit, sender_qubit)
    h(message_qubit)

    # Measure both of Alice's qubits. These two classical bits are the
    # *only* thing that leaves this function.
    sender_bit_0 = mz(message_qubit)
    sender_bit_1 = mz(sender_qubit)
    return [sender_bit_0, sender_bit_1]


@cudaq.kernel
def classical_channel_to_bob(sender_bit_0: bool, sender_bit_1: bool) -> list[bool]:
    """The physical transport step - and the *only* one in this whole
    protocol. Unlike superdense_coding.py's send_qubit_to_bob(), this
    is not standing in for a qubit crossing space: sender_bit_0 and
    sender_bit_1 are ordinary classical bits, sendable by phone, radio,
    or carrier pigeon with a note - anything slower than light is fine,
    which is exactly why teleportation doesn't violate relativity. It's
    still named and called explicitly, so "these 2 bits reached Bob"
    is a real step in the call sequence rather than something implied
    by bob_room() simply being handed the values."""
    return [sender_bit_0, sender_bit_1]


@cudaq.kernel
def bob_room(receiver_qubit: cudaq.qubit, sender_bit_0: bool, sender_bit_1: bool,
             measure_x_basis: bool) -> bool:
    """Bob's entire world: his half of the Bell pair - which has been
    sitting at his location since prepare_shared_bell_pair() and never
    physically moves again - plus the two classical bits that arrived
    via classical_channel_to_bob(). Never message_angle itself.
    Depending on which of the 4 possible outcomes Alice measured,
    receiver_qubit is currently the message state up to a known,
    fixable Pauli error; these corrections undo exactly that error."""
    if sender_bit_0:
        z(receiver_qubit)
    if sender_bit_1:
        x(receiver_qubit)

    # measure_x_basis is Bob's own choice of what to check, not a
    # secret from Alice - flips to the X basis before measuring so we
    # can verify the *full* state was moved, not just its Z statistics.
    if measure_x_basis:
        h(receiver_qubit)
    return mz(receiver_qubit)


@cudaq.kernel
def teleportation_protocol(message_angle: float, measure_x_basis: bool) -> bool:
    """The main driver: allocates all three qubits and calls each
    step in strict protocol order - shared setup, then Alice, then the
    classical transport, then Bob. This is the only place that "sees"
    both message_angle and the qubits, and even it never applies a
    gate directly."""
    qubits = cudaq.qvector(3)
    message_qubit = qubits[0]
    sender_qubit = qubits[1]
    receiver_qubit = qubits[2]

    prepare_shared_bell_pair(sender_qubit, receiver_qubit)
    sender_bits = alice_room(message_qubit, sender_qubit, message_angle)
    classical_bits = classical_channel_to_bob(sender_bits[0], sender_bits[1])
    return bob_room(receiver_qubit, classical_bits[0], classical_bits[1], measure_x_basis)


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
        z_bits = cudaq.run(teleportation_protocol, message_angle, False, shots_count=shots)
        x_bits = cudaq.run(teleportation_protocol, message_angle, True, shots_count=shots)
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
