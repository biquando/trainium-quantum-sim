import circuit
import numpy as np
import scipy
import time

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
    n = 20
    qc = circuit.QuantumCircuit(n, batch_size=512)
    qc.append(H, [0])
    for i in range(1, n):
        qc.append(CX, [0, i])

    state = qc.run(consolidate=False)
    print(state)

def test_from_qiskit():
    n = 10
    qqc = qiskit.QuantumCircuit(n)
    qqc.h(0)
    for i in range(1, n):
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
    print(f'Number of qubits: {n}')
    print(f'Gate size: {gate_size}')
    print(f'Number of gates: {NGATES}')

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
    start = time.time()
    kernel_state = qc.run()
    end = time.time()
    print('Runtime (s):', end - start)

    # Calculate the Euclidean distance between the two state vectors
    diff = kernel_state - qiskit_state[:, None]
    print('Distance between state vectors:', np.linalg.norm(diff))

def test_consolidation():
    n = 10
    gate_size = 2
    NGATES = 200

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
    qqc = qiskit.QuantumCircuit(n)
    for U, idcs in zip(unitaries, qubit_indices):
        qqc.append(UnitaryGate(U), list(idcs))
    qiskit_state = Statevector.from_instruction(qqc).data.astype(np.complex128)

    # Simulate circuit in Trainium
    qc = circuit.QuantumCircuit.from_qiskit(qqc)
    kernel_state = qc.run()
    print(f'Consolidated {NGATES} gates to {len(qc.gates)}')

    # Calculate the Euclidean distance between the two state vectors
    diff = kernel_state - qiskit_state[:, None]
    print('Distance between state vectors:', np.linalg.norm(diff))

def test_timing():
    n = 10
    gate_size = 7
    ngates = 5

    print(f'num qubits = {n}')
    print(f'num gates = {ngates}')

    unitary_mats = scipy.stats.unitary_group(dim=2**gate_size, seed=0)

    qc = circuit.QuantumCircuit(n, gate_size)
    for _ in range(ngates):
        U = unitary_mats.rvs().astype(np.complex128)
        idcs = np.random.choice(range(n), size=gate_size, replace=False)
        qc.append(U, idcs)

    state = qc.run(consolidate=False)
    print(state)



if __name__ == '__main__':
    print('\n=== Test: GHZ circuit from numpy unitaries ===')
    test_from_numpy()

    # print('\n=== Test: GHZ circuit imported from qiskit ===')
    # test_from_qiskit()

    # print('\n=== Test: GHZ circuit imported from qasm ===')
    # test_from_qasm()

    # print('\n=== Test: correctness (comparison with qiskit) ===')
    # test_correctness()

    # print('\n=== Test: consolidation ===')
    # test_consolidation()

    # print('\n=== Test: timing ===')
    # test_timing()
