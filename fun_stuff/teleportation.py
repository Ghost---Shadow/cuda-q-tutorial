"""
Minimal demo: why the Bell-pair step in quantum teleportation has to
happen BEFORE Alice's CNOT, not after.

See nvidia-cudaq-tutorials/07_quantum_teleportation.py for the full,
documented protocol. This file isolates one question about it: sender
and receiver get entangled once (Bell pair), then sender gets entangled
AGAIN with the message qubit. Doesn't that second entangling step
scramble/decohere the first one? And does the order of the two steps
even matter?

Answer, demonstrated below: the order matters a lot (swap it and
teleportation just stops working), but it isn't decoherence - nothing
is lost, the correlation structure just becomes 3-way instead of 2-way,
and that 3-way structure is exactly what makes teleportation possible.

qubits: message (Alice's unknown state), sender (Alice's half of the
Bell pair), receiver (Bob's half).

CORRECT order:
  1. h(sender); x.ctrl(sender, receiver)     <- Bell pair first
  2. ry(angle, message)
  3. x.ctrl(message, sender); h(message)     <- Alice's Bell-basis rotation
  4. measure message, sender; correct receiver

SWAPPED order (same 5 gates, steps 1 and 3 traded places):
  1. ry(angle, message)
  2. x.ctrl(message, sender); h(message)     <- done before sender is Bell-paired
  3. h(sender); x.ctrl(sender, receiver)
  4. measure message, sender; correct receiver

Both circuits use the identical 5 gates and the identical corrections -
only the position of the "Bell pair" block relative to the "Alice"
block changes.

A second question worth being precise about: the CORRECT-order circuit
has Alice run x.ctrl(message, sender) - a gate that never touches
receiver at all, receiver isn't even a parameter to that gate. So how
does information reach Bob's qubit? It's tempting to say the CNOT
"sends" something to receiver, but that would be exactly the
faster-than-light signaling Einstein/EPR/the no-communication theorem
rule out. It doesn't. What the CNOT changes is a JOINT property of the
3-qubit state (the correlations between message, sender, and
receiver's outcomes) - never a LOCAL property of receiver alone (its
own reduced state, i.e. what Bob sees if he measures only his own
qubit). before_bob_gets_the_bits() below measures receiver directly,
with zero correction applied - simulating Bob peeking at his qubit
before Alice's classical bits ever arrive. If real information had
reached receiver locally, this would track cos(angle) just like the
corrected version does. It doesn't: it stays flat at ~0 for every
angle, proving receiver's own statistics carry no information about
message_angle on their own. The message only becomes readable once
Bob has both his qubit AND the 2 classical bits - which, in the real
protocol, take the same one-way trip through spacetime as a phone call.
"""

import math

import cudaq


@cudaq.kernel
def teleport(message_angle: float, swap_order: bool) -> bool:
    # All three qubits start at |0>. Shorthand below: m = cos(angle/2),
    # s = sin(angle/2), so the message state ry(angle, message) creates
    # is m|0> + s|1> - see hello_cudaq.py for the same "State: ..."
    # comment style, tracking the joint state gate-by-gate.
    qubits = cudaq.qvector(3)
    message, sender, receiver = qubits[0], qubits[1], qubits[2]

    if swap_order:
        # --- BROKEN ORDER: message flows into sender before sender has
        # anywhere (receiver) to carry that information onward to. ---

        # ry prepares message in isolation - nothing else is entangled
        # yet. State: (m|0> + s|1>)_message |0>_sender |0>_receiver
        ry(message_angle, message)

        # message is CONTROL, sender is TARGET: message's undetermined
        # value gets copied into sender, permanently linking them.
        # State: m|00> + s|11>  (message,sender)  |0>_receiver
        x.ctrl(message, sender)

        # H on message folds that link into a form only recoverable by
        # measuring BOTH message and sender together (a Bell-basis
        # measurement) - but receiver isn't part of this correlation
        # at all yet, so there's nothing for the measurement to hand
        # off to it.
        h(message)

        # Only NOW does sender get Bell-paired with receiver - sender
        # is CONTROL, receiver is TARGET. Too late: sender's value is
        # already locked to message's, not free to carry it to
        # receiver. Receiver ends up correlated with message only
        # through sender as a fixed relay, not as the flexible "any
        # state at all" carrier teleportation needs (see the
        # column-by-column numbers this file prints - "swapped order"
        # never tracks cos(angle)).
        h(sender)
        x.ctrl(sender, receiver)
    else:
        # --- CORRECT ORDER: build the empty highway (Bell pair) before
        # putting anything on it. ---

        # H puts sender into superposition. State: (|0>+|1>)_sender |0>_receiver
        h(sender)

        # sender is CONTROL, receiver is TARGET: entangles the pair
        # sender will carry the message through, with the pair message
        # will eventually be teleported onto - established before
        # message even exists.
        # State: (|00> + |11>)_{sender,receiver}
        x.ctrl(sender, receiver)

        # Only now does the message qubit get its unknown state -
        # still totally disjoint from sender/receiver so far.
        # State: (m|0> + s|1>)_message  (|00> + |11>)_{sender,receiver}
        ry(message_angle, message)

        # message is CONTROL, sender is TARGET. This does NOT reach
        # receiver - receiver isn't a parameter to this gate and is
        # physically untouched by it. What it does is fold message's
        # value into sender, which is ALREADY entangled with receiver
        # from the Bell pair above - so this reshapes the JOINT
        # correlation between all 3 qubits' eventual measurement
        # outcomes. receiver's own LOCAL statistics (what Bob sees
        # measuring only his qubit, without Alice's classical bits)
        # are completely unaffected by this gate - see
        # before_bob_gets_the_bits() below for a direct measurement of
        # that claim, not just an assertion of it.
        x.ctrl(message, sender)

        # H on message completes the Bell-basis rotation. After this,
        # measuring message+sender leaves receiver holding the
        # original message state, up to one of 4 fixable Pauli errors
        # - the classic teleportation result.
        h(message)

    # Measuring collapses message and sender into 2 classical bits -
    # this is the ONLY thing that ever "leaves" this local circuit in
    # the real protocol (see nvidia-cudaq-tutorials/07_quantum_teleportation.py
    # for those 2 bits modeled as an explicit classical-channel step).
    bit0 = mz(message)
    bit1 = mz(sender)

    # bit0/bit1 classically control gates on receiver - this is where
    # the 2 classical bits "flow back in" to finish moving the state.
    if bit0:
        z(receiver)
    if bit1:
        x(receiver)
    return mz(receiver)


@cudaq.kernel
def teleport_with_partial_bits(message_angle: float, use_bit0: bool, use_bit1: bool,
                                measure_x_basis: bool) -> bool:
    """Same correct-order circuit, but Bob is only given ONE of Alice's
    2 bits (whichever use_bit0/use_bit1 says) - simulating "why not just
    send 1 bit instead of 2?" directly: he still applies whatever
    correction he can with what he's got, and ignores the other.

    Why 2 bits are structurally necessary, not just convenient: after
    Alice's Bell-basis rotation, the 4 equally-likely (bit0, bit1)
    outcomes leave receiver in 4 DIFFERENT states relative to the
    original message - identity, X-flipped, Z-flipped, or both X and Z.
    These are 4 genuinely distinct corrections (the single-qubit Pauli
    group mod phase), and nothing about bit0 lets you infer bit1 or
    vice versa - they're independent coin flips. Distinguishing 4
    possibilities takes 2 bits.

    IMPORTANT SUBTLETY, worth getting right rather than hand-waving:
    dropping the Z-correction (bit0) does NOT show up if you only ever
    measure in the Z basis - Z|psi> only flips a relative phase between
    |0> and |1>, which changes nothing about |amplitude|^2 in the Z
    basis. A missing Z-correction is invisible to <Z>, and only shows
    up in <X> (or any basis sensitive to phase). Symmetrically, a
    missing X-correction (bit1) shows up clearly in <Z> but leaves <X>
    close to correct. So: measure_x_basis lets the __main__ block check
    BOTH bases, which is the only way to see that BOTH bits are
    actually load-bearing - checking only one basis would make it look
    like one of the two bits doesn't matter, which is a measurement
    artifact, not a fact about the protocol.
    """
    qubits = cudaq.qvector(3)
    message, sender, receiver = qubits[0], qubits[1], qubits[2]

    h(sender)
    x.ctrl(sender, receiver)
    ry(message_angle, message)
    x.ctrl(message, sender)
    h(message)

    bit0 = mz(message)
    bit1 = mz(sender)

    if use_bit0 and bit0:
        z(receiver)
    if use_bit1 and bit1:
        x(receiver)
    if measure_x_basis:
        h(receiver)
    return mz(receiver)


@cudaq.kernel
def before_bob_gets_the_bits(message_angle: float) -> bool:
    """The correct-order circuit, but Bob measures receiver directly -
    no correction, because in the real protocol he hasn't received
    Alice's 2 classical bits yet at this point (they're still in
    transit). This models exactly what Bob would see if he peeked at
    his qubit before the classical channel delivers anything.

    message and sender still get measured (mz has to run for the
    circuit to be well-defined and to match the corrected version
    gate-for-gate up to the missing corrections) - but their bits are
    thrown away here, standing in for "not yet sent."
    """
    qubits = cudaq.qvector(3)
    message, sender, receiver = qubits[0], qubits[1], qubits[2]

    h(sender)
    x.ctrl(sender, receiver)
    ry(message_angle, message)
    x.ctrl(message, sender)
    h(message)

    mz(message)
    mz(sender)
    # No corrections applied - Bob reads receiver as-is.
    return mz(receiver)


def z_expectation(bits: list) -> float:
    return sum(-1 if b else 1 for b in bits) / len(bits)


def nearest_small_fraction_of_pi(angle: float, max_denominator: int = 12) -> tuple:
    """For sanity-checking that a chosen test angle isn't secretly a
    "nice" multiple of pi in disguise (e.g. 2.1 rad LOOKS generic but
    is only 0.0056 away from 2*pi/3, close enough that cos(2.1) prints
    as -0.5048 - easy to misread as "-0.5 plus shot noise" at a glance).
    Returns (numerator, denominator, how_close) for the closest
    n*pi/d among small d - "how_close" should be nowhere near 0 for a
    genuinely arbitrary angle."""
    best = None
    for d in range(1, max_denominator + 1):
        n = round(angle / (math.pi / d))
        candidate = n * math.pi / d
        distance = abs(angle - candidate)
        if best is None or distance < best[2]:
            best = (n, d, distance)
    return best


# GENERIC_ANGLE is deliberately NOT chosen to look "nice" in any sense
# - unlike 2.1 (which turns out to be uncomfortably close to 2*pi/3,
# see nearest_small_fraction_of_pi's printed check below), this value
# has no relationship to pi at all, so its cos()/sin() won't round to
# anything a reader could mistake for a small-denominator fraction.
GENERIC_ANGLE = 1.2345678
TEST_ANGLES = [0.0, math.pi / 3, math.pi / 2, GENERIC_ANGLE]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)
    shots = 4000

    print("=== Sanity check: is GENERIC_ANGLE secretly a 'nice' angle? ===")
    n, d, distance = nearest_small_fraction_of_pi(GENERIC_ANGLE)
    print(f"GENERIC_ANGLE = {GENERIC_ANGLE!r}")
    print(f"Closest small fraction of pi found: {n}*pi/{d} = {n * math.pi / d:.6f} "
          f"(distance {distance:.6f})")
    print("A genuinely arbitrary angle should land nowhere near any small")
    print("fraction - a distance this large (not ~0.005 like 2.1 rad was to")
    print("2*pi/3) means cos(GENERIC_ANGLE)/sin(GENERIC_ANGLE) can't be")
    print("mistaken for a rounded 'nice' number by coincidence.")

    print()
    print("angle       <Z> correct order  <Z> swapped order  <Z> expected")
    for angle in TEST_ANGLES:
        correct_bits = cudaq.run(teleport, angle, False, shots_count=shots)
        swapped_bits = cudaq.run(teleport, angle, True, shots_count=shots)
        correct_z = z_expectation(correct_bits)
        swapped_z = z_expectation(swapped_bits)
        expected_z = math.cos(angle)
        print(f"{angle:9.6f}  {correct_z:16.6f}  {swapped_z:17.6f}  {expected_z:12.6f}")

    print()
    print("Expected: the 'correct order' column tracks cos(angle) closely")
    print("(within shot noise) for every angle - the message survives.")
    print("The 'swapped order' column does NOT track cos(angle) - it stays")
    print("roughly flat regardless of angle, because by the time sender")
    print("gets Bell-paired with receiver, sender has already been")
    print("collapsed into a fixed correlation with message that the later")
    print("Bell-pair step can't undo. Nothing 'decohered' in the lossy")
    print("sense - every gate ran, every qubit is still fully coherent -")
    print("the circuit is just a different, non-teleporting circuit.")

    print()
    print("=== No-signaling check: what does Bob see BEFORE the 2")
    print("    classical bits arrive? ===")
    print("angle       <Z> with correction  <Z> WITHOUT correction  <Z> expected (no signal = 0)")
    for angle in TEST_ANGLES:
        corrected_bits = cudaq.run(teleport, angle, False, shots_count=shots)
        uncorrected_bits = cudaq.run(before_bob_gets_the_bits, angle, shots_count=shots)
        corrected_z = z_expectation(corrected_bits)
        uncorrected_z = z_expectation(uncorrected_bits)
        print(f"{angle:9.6f}  {corrected_z:18.6f}  {uncorrected_z:21.6f}  {'0.000000':>14}")

    print()
    print("Expected: 'with correction' reproduces the cos(angle) curve from")
    print("the table above. 'WITHOUT correction' stays pinned near 0.0000 for")
    print("every angle, proving Bob's own qubit statistics carry zero")
    print("information about message_angle until he combines them with")
    print("Alice's classical bits. x.ctrl(message, sender) changed a JOINT")
    print("correlation across all 3 qubits, not receiver's own local state -")
    print("which is exactly what the no-communication theorem requires: no")
    print("gate applied to message/sender alone can let Bob learn anything by")
    print("looking at receiver in isolation, no matter how entangled it is.")

    print()
    print("=== Why 2 bits, not 1? Give Bob only one of Alice's bits, ===")
    print("=== and check BOTH measurement bases, not just one ===")
    print("angle       <Z> both  <Z> bit0-only  <Z> bit1-only  |  <X> both  <X> bit0-only  <X> bit1-only")
    for angle in TEST_ANGLES:
        z_both = cudaq.run(teleport_with_partial_bits, angle, True, True, False, shots_count=shots)
        z_bit0 = cudaq.run(teleport_with_partial_bits, angle, True, False, False, shots_count=shots)
        z_bit1 = cudaq.run(teleport_with_partial_bits, angle, False, True, False, shots_count=shots)
        x_both = cudaq.run(teleport_with_partial_bits, angle, True, True, True, shots_count=shots)
        x_bit0 = cudaq.run(teleport_with_partial_bits, angle, True, False, True, shots_count=shots)
        x_bit1 = cudaq.run(teleport_with_partial_bits, angle, False, True, True, shots_count=shots)
        print(f"{angle:9.6f}  {z_expectation(z_both):8.6f}  {z_expectation(z_bit0):12.6f}  "
              f"{z_expectation(z_bit1):12.6f}  |  {z_expectation(x_both):8.6f}  "
              f"{z_expectation(x_bit0):12.6f}  {z_expectation(x_bit1):12.6f}")
    print(f"expected      {'cos(angle)':^37}  |  {'sin(angle)':^37}")

    print()
    print("Expected: 'both' tracks cos(angle)/sin(angle) in both columns -")
    print("the only fully correct case. 'bit1-only' (X-correction applied,")
    print("Z-correction skipped) tracks <Z> FINE but is visibly off on <X> -")
    print("a missing Z-correction is a pure phase error, invisible to a Z-basis")
    print("measurement but not to an X-basis one. 'bit0-only' (Z-correction")
    print("applied, X-correction skipped) shows the opposite pattern: close on")
    print("<X>, visibly wrong on <Z>. Neither single bit is ever enough for")
    print("BOTH bases at once - only checking one basis would wrongly suggest")
    print("one of the two bits doesn't matter. There are 4 equally-likely,")
    print("mutually exclusive corrections Bob might need (I, X, Z, XZ);")
    print("distinguishing 4 possibilities takes log2(4) = 2 bits, full stop -")
    print("1 bit is information-theoretically insufficient, not just")
    print("inconvenient, and this is why: it's provably blind to half the")
    print("possible errors, in whichever basis you happen to check.")
