import cudaq


@cudaq.kernel
def bell_pair():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])
    mz(qubits)


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    print("Target:", cudaq.get_target().name)

    result = cudaq.sample(bell_pair, shots_count=1000)
    print("Measurement results:")
    print(result)
