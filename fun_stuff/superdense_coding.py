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
"""

import cudaq


@cudaq.kernel
def superdense_coding(bit0: bool, bit1: bool):
    """Encodes (bit0, bit1) into a Bell pair via Alice's local gates,
    then decodes it via Bob's disentangling circuit. Returns both
    measured bits - they should exactly match the inputs, every shot,
    with zero noise on an ideal simulator."""
    qubits = cudaq.qvector(2)

    # Shared entangled resource: qubits[0] is Alice's, qubits[1] is Bob's.
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])

    # Alice encodes her 2 classical bits onto her qubit alone. This is
    # the only step Alice performs, and it never touches qubits[1].
    if bit1:
        x(qubits[0])
    if bit0:
        z(qubits[0])

    # Alice physically sends qubits[0] to Bob. In simulation there's
    # nothing to "send" - Bob just gets to act on both qubits now.

    # Bob's decoding circuit: exactly undoes the Bell-pair preparation.
    x.ctrl(qubits[0], qubits[1])
    h(qubits[0])

    mz(qubits[0])
    mz(qubits[1])


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(7)
    shots = 1000

    print("bit0  bit1  ->  measured0  measured1  match")
    all_match = True
    for bit0 in (False, True):
        for bit1 in (False, True):
            results = cudaq.sample(superdense_coding, bit0, bit1, shots_count=shots)
            # An ideal run collapses to a single deterministic outcome -
            # every shot should agree, so there should be exactly one
            # bitstring in the histogram.
            outcome = max(results.items(), key=lambda kv: kv[1])[0]
            measured0, measured1 = outcome[0] == "1", outcome[1] == "1"
            match = (measured0 == bit0) and (measured1 == bit1) and len(results) == 1
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
