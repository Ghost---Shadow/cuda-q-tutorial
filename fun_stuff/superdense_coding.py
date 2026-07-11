"""
Superdense Coding

The mirror image of quantum teleportation (see
nvidia-cudaq-tutorials/07_quantum_teleportation.py): teleportation moves
1 unknown qubit using a shared Bell pair + 2 classical bits. Superdense
coding moves 2 known classical bits using a shared Bell pair + 1 qubit.

Alice wants to send Bob one of 4 classical messages: 00, 01, 10, 11.
They pre-share a Bell pair (qubits[0] with Alice, qubits[1] with Bob).
Alice applies one of 4 gates to *her* qubit alone, based on her 2 bits,
then sends that single qubit to Bob. Bob, now holding both halves of
the pair, applies CNOT then Hadamard (undoing the Bell-pair prep) and
measures both qubits straight out - recovering Alice's 2 bits exactly.

The trick: Alice's local gate doesn't "add information" to her qubit in
any classical sense - a single qubit can only ever yield 1 classical bit
under measurement. What her gate does is rotate the *joint* two-qubit
state into one of the 4 mutually distinguishable Bell states. Only Bob,
holding both halves, can tell those 4 states apart - which is why he
still needs Alice's one physical qubit to arrive before he can decode
anything. No information travels faster than light: the classical
channel here is the qubit itself being physically transported.

Bit-to-gate mapping (the 4 Bell states):
  00 -> I          : |Phi+> = (|00> + |11>)/sqrt(2)
  01 -> X          : |Psi+> = (|01> + |10>)/sqrt(2)
  10 -> Z          : |Phi-> = (|00> - |11>)/sqrt(2)
  11 -> X then Z   : |Psi-> = (|01> - |10>)/sqrt(2)

Code structure is deliberately "nothing up my sleeve": alice_room() is
the *only* function that ever sees bit0/bit1, and it only ever touches
alice_qubit - it has no parameter through which bob_qubit's state could
leak in, and no return value through which bit0/bit1 could leak out.
bob_room() is the *only* function that measures anything, and its
signature has no bit0/bit1 parameter at all - it's physically incapable
of reading Alice's bits directly out of Python state; the only channel
between the two rooms is the shared qubit pair itself, exactly like the
real protocol.

The physical transport step is made explicit too, not just implied by
call order: send_qubit_to_bob() marks the exact moment alice_qubit
stops being local to Alice and becomes local to Bob. On a single
simulator there's no second machine to actually move it to, so the
function body is a no-op - but it's a real, separately-named step in
the protocol, not something silently absorbed into bob_room() calling
gates on a qubit it was never actually handed. Contrast with
07_quantum_teleportation.py, where the corresponding step
(classical_channel_to_bob) sends 2 classical bits instead - no qubit
is ever transported there at all.
"""

import cudaq


@cudaq.kernel
def prepare_shared_bell_pair(alice_qubit: cudaq.qubit, bob_qubit: cudaq.qubit):
    """The trusted setup phase, run before Alice or Bob do anything: a
    Bell pair, split so Alice keeps one half and Bob keeps the other.

    Same "State: ..." tracking style as hello_cudaq.py's bell_pair()."""
    # Both start at |0>. State: |0>_alice |0>_bob
    h(alice_qubit)
    # alice_qubit now undetermined. State: (|0>+|1>)_alice |0>_bob
    x.ctrl(alice_qubit, bob_qubit)
    # alice_qubit is CONTROL, bob_qubit is TARGET - entangles them.
    # State: (|00> + |11>)_{alice,bob}   (unnormalized)


@cudaq.kernel
def alice_room(alice_qubit: cudaq.qubit, bit0: bool, bit1: bool):
    """Alice's entire world. She has her own qubit and her own 2 bits -
    nothing else. She cannot see bob_qubit (it isn't a parameter here),
    so there is no way for this function to accidentally act on it.

    Entering this function, the joint state is still
    (|00> + |11>)_{alice,bob} from prepare_shared_bell_pair() - Alice
    only ever touches her own half of it, but because that half is
    entangled with bob_qubit, every gate below reshapes the FULL
    2-qubit state even though bob_qubit never appears in this
    function's code at all."""
    if bit1:
        x(alice_qubit)
        # X on alice_qubit swaps the |0>/|1> label on her half.
        # State: (|10> + |01>)_{alice,bob} - the |Psi+> Bell state.
    if bit0:
        z(alice_qubit)
        # Z on alice_qubit flips the sign of wherever alice_qubit=|1>.
        # bit0=1,bit1=0: (|00> - |11>)  -> |Phi->
        # bit0=1,bit1=1: (|10> - |01>)  -> |Psi->
    # bit0=0,bit1=0 (no gates ran): state stays (|00> + |11>) -> |Phi+>
    # Four (bit0,bit1) inputs, four mutually distinguishable joint
    # states - this is the entire information capacity Alice is using.


@cudaq.kernel
def send_qubit_to_bob(alice_qubit: cudaq.qubit):
    """The physical transport step: Alice hands her one qubit - and
    only that qubit, never bob_qubit - over to Bob. This is a real,
    physically-required step of the protocol (a photon down a fiber, a
    trapped ion physically carried over, whatever the hardware is) -
    it just happens to be a no-op in code, since a single simulator
    process has nowhere else to move the qubit to. It's named and
    called explicitly anyway, so "Bob has the qubit now" is a fact
    that appears in the protocol's call sequence, not something you'd
    have to infer from bob_room() simply being allowed to touch
    alice_qubit."""
    pass


@cudaq.kernel
def bob_room(alice_qubit: cudaq.qubit, bob_qubit: cudaq.qubit) -> list[bool]:
    """Bob's entire world. He receives Alice's physical qubit (the
    *only* thing that crosses from her room to his, and only after
    send_qubit_to_bob() has run) plus his own half of the pair - never
    bit0/bit1 directly. Everything he learns, he learns by measuring."""
    # One of the 4 Bell states walks in (see alice_room's comments for
    # which one, based on bit0/bit1) - Bob doesn't know which, that's
    # exactly what he's about to find out.
    x.ctrl(alice_qubit, bob_qubit)
    # alice_qubit is CONTROL, bob_qubit is TARGET - the inverse of the
    # CNOT that built the pair. This un-mixes the two qubits back
    # toward a product state, but only completely for bit1=0 inputs;
    # bit1=1 inputs need the H below too.
    h(alice_qubit)
    # H on alice_qubit finishes rotating each of the 4 Bell states back
    # to a distinct, unentangled computational-basis state: |00>, |01>,
    # |10>, or |11> - exactly encoding (bit0, bit1).
    measured0 = mz(alice_qubit)
    measured1 = mz(bob_qubit)
    return [measured0, measured1]


@cudaq.kernel
def superdense_coding_protocol(bit0: bool, bit1: bool) -> list[bool]:
    """The main driver: allocates the shared qubit pair and calls each
    room's function in strict protocol order - setup, then Alice, then
    the physical qubit transport, then Bob. This is the only place
    that "sees" both halves of the story, and even it never touches a
    gate directly."""
    qubits = cudaq.qvector(2)
    alice_qubit = qubits[0]
    bob_qubit = qubits[1]

    prepare_shared_bell_pair(alice_qubit, bob_qubit)
    alice_room(alice_qubit, bit0, bit1)
    send_qubit_to_bob(alice_qubit)
    return bob_room(alice_qubit, bob_qubit)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)
    shots = 1000

    print("bit0  bit1  ->  measured0  measured1  match")
    all_match = True
    for bit0 in (False, True):
        for bit1 in (False, True):
            outcomes = cudaq.run(superdense_coding_protocol, bit0, bit1, shots_count=shots)
            # An ideal run is fully deterministic - every shot should
            # decode to the exact same pair of bits Alice sent.
            unique_outcomes = {tuple(outcome) for outcome in outcomes}
            measured0, measured1 = outcomes[0]
            match = (measured0 == bit0) and (measured1 == bit1) and len(unique_outcomes) == 1
            all_match = all_match and match
            print(f"{int(bit0):>4}  {int(bit1):>4}  ->  {int(measured0):>9}  "
                  f"{int(measured1):>9}  {match}")

    print()
    if all_match:
        print("All 4 classical messages recovered exactly, every shot: "
              "1 physically-transmitted qubit + 1 pre-shared entangled "
              "qubit successfully carried 2 classical bits.")
    else:
        print("MISMATCH - superdense coding did not decode correctly.")
