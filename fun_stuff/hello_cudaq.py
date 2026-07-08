import cudaq


@cudaq.kernel
def apply_hadamard(q: cudaq.qubit):
    h(q)


@cudaq.kernel
def apply_cnot(control: cudaq.qubit, target: cudaq.qubit):
    x.ctrl(control, target)


@cudaq.kernel
def measure_all(qubits: cudaq.qview):
    mz(qubits)


@cudaq.kernel
def bell_pair():
    # Both qubits start in |0>. State: |0> , |0>
    qubits = cudaq.qvector(2)

    # Hadamard puts qubit 0 into an equal superposition of |0> and |1>.
    # State: (|0> + |1>) , |0>   [qubit 0 undetermined, qubit 1 still |0>]
    apply_hadamard(qubits[0])

    # CNOT entangles qubit 1 with qubit 0: if qubit 0 is |1>, flip qubit 1.
    # State: |00> + |11>   (unnormalized Bell state)
    # This "+" is a weighted sum of amplitudes, not a classical OR - both
    # qubits stay undetermined (no pre-existing value for either one) until
    # mz collapses them, 50/50, to one matching pair or the other.
    apply_cnot(qubits[0], qubits[1])

    measure_all(qubits)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    print("Target:", cudaq.get_target().name)

    # Fix the RNG seed so sample() produces the same shot-by-shot outcomes
    # (and therefore the same histogram) on every run.
    cudaq.set_random_seed(42)

    # Expected output: only "00" and "11" should ever appear, each roughly
    # 50% of the 1000 shots (e.g. { 00:478 11:522 }). "01" and "10" should
    # never appear, since the Bell state has zero probability of the qubits
    # disagreeing - that's the signature of entanglement.
    result = cudaq.sample(bell_pair, shots_count=1000)
    print("Measurement results:")
    print(result)

    # How to read the printed histogram, e.g. { 00:470 11:530 }:
    # - It's a bitstring:count map, one entry per distinct outcome observed
    #   across all 1000 shots. The two characters in each bitstring are the
    #   measured values of qubits[0] and qubits[1], in that order.
    # - "00:470" means both qubits were measured as 0 on 470 of the 1000
    #   runs; "11:530" means both were measured as 1 on the other 530 runs.
    # - 470/530 is a normal statistical split around the ideal 50/50 - each
    #   shot's outcome is genuinely random, only the *joint* outcome is
    #   fixed (always matching). More shots would push the ratio closer to
    #   50/50; a fixed 1000-shot run will always show some sampling noise.
    # - If you ever saw "01" or "10" in this histogram, that would indicate
    #   either a bug in the circuit or a noisy/decohered qubit, since the
    #   ideal Bell state assigns them zero probability.
