from qiskit import QuantumCircuit
import numpy as np
from qiskit.circuit.library import UnitaryGate

def kron(*args) -> np.ndarray:
    mat = np.array([[1]])
    for arg in args:
        mat = np.kron(mat, arg)
    return mat

def kron_pad(U, left_bits, right_bits) -> np.ndarray:
    left_bits = max(left_bits, 0)
    right_bits = max(right_bits, 0)
    return kron(
        np.identity(2**left_bits),
        U,
        np.identity(2**right_bits)
    )


def consolidate(n: int, gate_size: int, gates: list[tuple[np.ndarray, list[int]]]):
    assert gate_size <= n

    gates = []

    curr_gate = QuantumCircuit(gate_size)
    curr_idcs = []
    curr_idcs_inv = {}

    for U, idcs in gates:
        if set(idcs) <= set(curr_idcs):
            # we can consolidate in the current gate
            curr_gate.append(UnitaryGate(U), [curr_idcs_inv[i] for i in idcs])
        elif len(set(idcs) - set(curr_idcs)) <= gate_size - len(curr_idcs):
            # we can add to the current gate
            for idx in idcs:
                if idx not in curr_idcs:
                    curr_idcs_inv[idx] = len(curr_idcs)
                    curr_idcs.append(idx)
            curr_gate.append(UnitaryGate(U), [curr_idcs_inv[i] for i in idcs])
        else:
            # we must flush the current gate
            for i in range(n):
                if len(curr_idcs) == gate_size:
                    break
                if i in curr_idcs:
                    continue
                curr_idcs.append(i)
            assert len(curr_idcs) == gate_size
            qc.append(curr_gate, curr_idcs)
            curr_idcs = []
            curr_idcs_inv = {}

            for idx in idcs:
                if idx not in curr_idcs:
                    curr_idcs_inv[idx] = len(curr_idcs)
                    curr_idcs.append(idx)
            curr_gate.append(UnitaryGate(U), [curr_idcs_inv[i] for i in idcs])

    # flush last gate
    for i in range(n):
        if len(curr_idcs) == gate_size:
            break
        if i in curr_idcs:
            continue
        curr_idcs.append(i)
    assert len(curr_idcs) == gate_size
    qc.append(curr_gate, curr_idcs)

    # output gates
    gates = []
    for gate in qc.data:
        U = gate.matrix
        idcs = [qubit._index for qubit in gate.qubits]
        assert len(idcs) <= gate_size
        gates.append((U, idcs))
    return gates
