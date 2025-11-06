import math
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZGate, UnitaryGate, QFT
from qiskit.quantum_info import Operator
from qiskit_aer import StatevectorSimulator
import numpy as np

def consolidate(gate_size, n, gates):
    new_gates = []

    curr_gate = qiskit.QuantumCircuit(gate_size)
    curr_idcs = []
    curr_idcs_inv = {}

    # Consolidate the gate described by (U, idcs) into curr_gate
    def add_to_gate(U, idcs):
        for idx in idcs:
            if idx not in curr_idcs:
                curr_idcs_inv[idx] = len(curr_idcs)
                curr_idcs.append(idx)
        assert len(curr_idcs) <= gate_size
        curr_gate.append(UnitaryGate(U), [curr_idcs_inv[i] for i in idcs])

    # Add curr_gate to the output list of gates, and reset it back to identity
    def flush_gate():
        nonlocal curr_idcs, curr_idcs_inv, curr_gate
        # print(f'consolidating {len(curr_gate)} gates')

        # Add qubits until the curr_gate size reaches self.gate_size
        for i in range(n):
            if len(curr_idcs) == gate_size:
                break
            if i in curr_idcs:
                continue
            curr_idcs.append(i)
        assert len(curr_idcs) == gate_size

        # Convert curr_gate (a qiskit circuit) to a unitary matrix
        new_gates.append((Operator(curr_gate).data, curr_idcs))

        curr_gate = qiskit.QuantumCircuit(gate_size)
        curr_idcs = []
        curr_idcs_inv = {}

    for U, idcs in gates:
        can_consolidate = len(set(idcs) - set(curr_idcs)) <= gate_size - len(curr_idcs)
        if not can_consolidate:
            flush_gate()
        add_to_gate(U, idcs)

    flush_gate()

    return new_gates



for n in range(16, 28):
    # print(f'\n\n=====================================')
    # print(f'============== n = {n} ==============')
    # print(f'=====================================')
    print(f'\n=== {n = } ===')
    # n = 25
    gate_size = 7

    qc = QFT(n)
    # qc = QuantumCircuit(n)
    # target_state = '0'*(n-1)
    # n_iters = round(math.pi/4 * math.sqrt(2**n))
    # for _ in range(n_iters):
    #     qc.append(ZGate().control(n-1, ctrl_state=target_state), range(n))
    #     for i in range(n-1): qc.h(i)
    #     qc.append(ZGate().control(n-1, ctrl_state='0'*(n-1)), range(n))
    #     for i in range(n-1): qc.h(i)
    # print(len(qc.data))
    # exit(0)

    tqc = transpile(qc, basis_gates=['u', 'cx'])
    # print(tqc.draw())


    gates = []
    for gate in tqc.data:
        gates.append((gate.matrix, [idx._index for idx in gate[1]]))
        # gates.append((gate.matrix, np.random.choice(range(n), size=len(gate[1]), replace=False)))
    new_gates = consolidate(gate_size, n, gates)

    new_qc = QuantumCircuit(n)
    for U, idcs in new_gates:
        # idcs2 = [x._index for x in idcs if type(x) is not int]
        # print(idcs2)
        if len(idcs) < gate_size:
            idcs += list(range(len(idcs), gate_size))
        new_qc.append(UnitaryGate(U), idcs)

    num_orig_gates = len(tqc.data)
    num_new_gates = len(new_gates)
    # print(f'{n_iters=}')
    # print(f'{num_orig_gates=}')
    # print(f'{num_new_gates=}')
    # print(f'ratio={num_orig_gates/num_new_gates}')


    # print('\n============== Default circuit ==============')
    backend = StatevectorSimulator()
    job = backend.run(tqc, shots=1)
    print('default gate time:', job.result().results[0].metadata['time_taken'] / len(tqc.data))
    # print(job.result())

    # print('\n============== No parallelization ==============')
    backend = StatevectorSimulator(max_parallel_threads=1)
    job = backend.run(tqc, shots=1)
    print('no parallel gate time:', job.result().results[0].metadata['time_taken'] / len(tqc.data))
    # print(job.result())

    # print('\n============== Consolidated gates ==============')
    backend = StatevectorSimulator(fusion_enable=False)
    job = backend.run(new_qc, shots=1)
    print('consolidate src gate time:', job.result().results[0].metadata['time_taken'] / len(tqc.data))
    print('consolidate gate time:', job.result().results[0].metadata['time_taken'] / len(new_gates))
    # print(job.result())

