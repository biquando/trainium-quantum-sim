import circuit
import numpy as np
import scipy

import qiskit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

H = np.array([
    [1,  1],
    [1, -1],
], dtype=np.complex128) / np.sqrt(2)

X = np.array([
    [0,  1],
    [1,  0],
], dtype=np.complex128)

CX = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
], dtype=np.complex128)


def test_from_numpy():
    qc = circuit.QuantumCircuit(10)
    qc.append(H, [0])
    for i in range(1, 10):
        qc.append(CX, [0, i])

    state = qc.run()
    print(state)

def test_from_qiskit():
    qqc = qiskit.QuantumCircuit(10)
    qqc.h(0)
    for i in range(1, 10):
        qqc.cx(0, i)
    qc = circuit.QuantumCircuit.from_qiskit(qqc)

    state = qc.run()
    print(state)

def test_from_qasm():
    qc = circuit.QuantumCircuit.from_qasm_str("""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[10];
    h q[0];
    cx q[0], q[1];
    cx q[0], q[2];
    cx q[0], q[3];
    cx q[0], q[4];
    cx q[0], q[5];
    cx q[0], q[6];
    cx q[0], q[7];
    cx q[0], q[8];
    cx q[0], q[9];
    """)

    state = qc.run()
    print(state)

def test_correctness():
    n = 10
    gate_size = 7
    NGATES = 20

    np.random.seed(0)
    unitary_mats = scipy.stats.unitary_group(dim=2**gate_size, seed=0)

    # Generate some random gates
    unitaries = []
    qubit_indices = []
    for _ in range(NGATES):
        U = unitary_mats.rvs().astype(np.complex128)
        unitaries.append(U)

        idcs = np.random.choice(range(n), size=gate_size, replace=False)
        qubit_indices.append(idcs)

    # Simulate circuit in Qiskit
    qc = qiskit.QuantumCircuit(n)
    for U, idcs in zip(unitaries, qubit_indices):
        qc.append(UnitaryGate(U), list(idcs))
    qiskit_state = Statevector.from_instruction(qc).data.astype(np.complex128)

    # Simulate circuit in Trainium
    qc = circuit.QuantumCircuit.from_qiskit(qc)
    kernel_state = qc.run()

    # Calculate the Euclidean distance between the two state vectors
    diff = kernel_state - qiskit_state[:, None]
    print('Distance between state vectors:', np.linalg.norm(diff))


if __name__ == '__main__':
    print('\n=== Test: GHZ circuit from numpy unitaries ===')
    test_from_numpy()

    print('\n=== Test: GHZ circuit from qiskit ===')
    test_from_qiskit()

    print('\n=== Test: GHZ circuit from qasm ===')
    test_from_qasm()

    print('\n=== Test: correctness (comparison with qiskit) ===')
    test_correctness()
