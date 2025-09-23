from __future__ import annotations
import numpy as np
import time
import torch_circuit

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import torch
import torch_xla.core.xla_model as xm

import qiskit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


def _kron(*args) -> np.ndarray:
    mat = np.array([[1]])
    for arg in args:
        mat = np.kron(mat, arg)
    return mat

def _kron_pad(U, left_bits, right_bits) -> np.ndarray:
    left_bits = max(left_bits, 0)
    right_bits = max(right_bits, 0)
    return _kron(
        np.identity(2**left_bits),
        U,
        np.identity(2**right_bits)
    )


class QuantumCircuit:
    def __init__(self, n_qubits: int, max_gate_size: int = 7, batch_size: int = 512, dtype=np.float32):
        self.n = n_qubits                                  # number of qubits
        self.N = int(2 ** self.n)                          # length of state vector
        self.gate_size = min(max_gate_size, n_qubits)      # number of qubits per gate
        self.unitary_size = int(2 ** self.gate_size)       # size of each gate's unitary
        self.ntiles = int(2 ** (self.n - self.gate_size))  # number of tiles that we must multiply

        assert batch_size <= 512
        self.batch_size = batch_size
        self.dtype = dtype

        # We store gates as a list of tuples with two elements:
        #  1. complex-valued unitary matrix
        #  2. qubit indices that the gate acts on
        # TODO: we can consolidate gates in this representation
        self.gates: list[tuple[np.ndarray, list[int]]] = []

    # Append a gate to the circuit.
    # Params:
    #  - U: unitary matrix, on at most self.gate_size qubits, type=np.complex128
    #  - qubit_idcs: list of qubits that the gate acts on
    def append(self, U: np.ndarray, qubit_idcs: list[int]) -> None:
        # Calculate the number of qubits that U acts on
        U_nqubits = int(np.floor(np.log2(U.shape[0])))

        # Validate U
        assert len(U.shape) == 2 and U.shape[0] == U.shape[1]   # square matrix
        assert 2 ** U_nqubits == U.shape[0]                     # valid 2^m dimensions
        assert U_nqubits <= self.gate_size                      # below max gate size
        assert U_nqubits == len(qubit_idcs)                     # correct number of qubits

        self.gates.append((U, qubit_idcs))

    # Import gates from a qiskit.QuantumCircuit
    @staticmethod
    def from_qiskit(qiskit_qc: qiskit.QuantumCircuit) -> QuantumCircuit:
        qc = QuantumCircuit(len(qiskit_qc.qubits))

        for gate in qiskit_qc.data:
            U = gate.matrix
            idcs = [qubit._index for qubit in gate.qubits]
            assert len(idcs) <= qc.gate_size

            qc.append(U, idcs)

        return qc

    # Import gates from a qasm string
    @staticmethod
    def from_qasm_str(qasm: str) -> QuantumCircuit:
        qqc = qiskit.QuantumCircuit.from_qasm_str(qasm)
        return QuantumCircuit.from_qiskit(qqc)


    # Simulate the circuit with initial state |0>.
    # Returns:
    #    final state vector of shape (N, 1) with type np.complex128
    def run(self, consolidate=True) -> np.ndarray:
        if consolidate:
            self.consolidate()

        # Pre-process gates to match the following format (tuples of three elements):
        #  1. real part of NxN unitary
        #  2. imaginary part of NxN unitary
        #  3. permutation information from QuantumCircuit.get_perm()
        kernel_gates: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
        for U, qubit_idcs in self.gates:
            # If U is less than self.gate_size, than we pad it with identities
            assert len(qubit_idcs) <= self.gate_size
            U = _kron_pad(U, self.gate_size - len(qubit_idcs), 0)

            kernel_gates.append((
                U.real.astype(self.dtype),
                U.imag.astype(self.dtype),
                QuantumCircuit.get_perm(self.n, qubit_idcs),
            ))

        with torch.no_grad():
            return self._run_model(kernel_gates)

    def _run_model(self, kernel_gates):

        device = xm.xla_device()
        device = 'xla'

        # Create model
        start = time.time()
        base = np.array(
            range(self.unitary_size * self.batch_size),
            dtype=np.uint32
        ).reshape((self.batch_size, self.unitary_size)).T

        qc = torch_circuit.QC(
            self.N, self.unitary_size, self.ntiles, self.batch_size, base
        ).to(device)
        end = time.time()
        print(f'time to create model (s): {end - start}')

        # Send gates to device
        start = time.time()
        unitaries = torch.stack([
            torch.stack([torch.from_numpy(Ur), torch.from_numpy(Ui)])
            for Ur, Ui, _ in kernel_gates
        ]).to(device)

        idcs = torch.stack([
            torch.from_numpy(i)
            for _, _, i in kernel_gates
        ]).to(device)
        end = time.time()
        print(f'time to send gates to device (s): {end - start}')

        # # Initialize state
        # start = time.time()
        # # FIXME: dtype should come from the class
        # initial_state = torch.zeros((2, self.N, 1), dtype=torch.float32, device=device)
        # initial_state[0, 0, 0] = 1.0
        # end = time.time()
        # print(f'time to initialize state (s): {end - start}')

        start = time.time()
        # state = QuantumCircuit.kernel(kernel_gates,
        #                               self.N,
        #                               self.unitary_size,
        #                               self.ntiles,
        #                               self.dtype,
        #                               self.batch_size,
        #                               base)
        print(unitaries.shape)
        print(idcs.shape)
        # print(initial_state.shape)
        # state = qc(unitaries, idcs, initial_state)
        state = qc(unitaries, idcs)

        state = state.to('cpu').numpy()
        end = time.time()
        print(f'number of qubits: {self.n}')
        print(f'number of gates: {len(kernel_gates)}')
        print(f'kernel time (s): {end - start}')

        # Combine real and imaginary components
        state = state[0].astype(np.complex128) \
              + state[1].astype(np.complex128) * 1.0j
        return state


    # Input gates:
    #  1. real part of NxN unitary
    #  2. imaginary part of NxN unitary
    #  3. qubit permutation information from QuantumCircuit.get_perm()
    # returns: final state vector with separated real/imag components, shape=(2, N, 1)
    @staticmethod
    @nki.compiler.skip_middle_end_transformations
    @nki.jit
    # @nki.profile(working_directory="/home/ubuntu/profiles", save_neff_name='file.neff', save_trace_name='profile.ntff')
    def kernel(unitaries, idcs, N, unitary_size, ntiles, batch_size, base):
        # Initialize state vector to |0>
        state = nl.ndarray((2, N, 1), dtype=unitaries.dtype, buffer=nl.shared_hbm)
        for tile_idx in nl.affine_range(ntiles):
            nl.store(dst=state[0, tile_idx * unitary_size : (tile_idx+1) * unitary_size, 0], value=0.0)
            nl.store(dst=state[1, tile_idx * unitary_size : (tile_idx+1) * unitary_size, 0], value=0.0)
        nl.store(dst=state[0, 0, 0], value=1.0)

        # This just stores [0, 1, 2, ..., 127], for use in indexing
        # tile_idcs_base_range = nl.arange(unitary_size)[:, None]
        # tile_idcs_base = nisa.iota(tile_idcs_base_range, dtype=np.uint32)
        tile_idcs_base = nl.load(base)

        for gate_idx in nl.affine_range(unitaries.shape[0]):
            gate = (unitaries[gate_idx][0], unitaries[gate_idx][1], idcs[gate_idx])
            assert type(gate) == tuple and len(gate) == 3
            assert gate[0].shape == (unitary_size, unitary_size)
            assert gate[1].shape == (unitary_size, unitary_size)
            assert gate[2].shape[1] == 4

            # This is the 128x128 matrix corresponding the current gate
            U_tile_real = nl.load(gate[0])
            U_tile_imag = nl.load(gate[1])

            # We apply the matrix to each chunk of the state vector
            for tile_idx in nl.affine_range(ntiles // batch_size): # allows for parallel computation
                offset = tile_idx * unitary_size * batch_size
                tile_idcs = nl.add(tile_idcs_base, offset, dtype=np.uint32)

                # Convert the qubit permutation to a statevector permutation using bit operations
                y = nl.copy(tile_idcs) # permuted tile idcs
                for i in nl.sequential_range(gate[2].shape[0]):
                    params = nl.load(gate[2][i])
                    a  = nl.load(gate[2][i][0])
                    b  = nl.load(gate[2][i][1])
                    ma = nl.load(gate[2][i][2]) # 2^a
                    mb = nl.load(gate[2][i][3]) # 2^b

                    y[:] = nl.bitwise_and(y, nl.invert(mb))
                    y[:] = nl.bitwise_or(y, nl.left_shift(nl.bitwise_and(tile_idcs, ma), b - a))
                    y[:] = nl.bitwise_or(y, nl.right_shift(nl.bitwise_and(tile_idcs, ma), a - b))


                # *** Load/Multiply/Store (3M) ***

                # state_tile_real = nl.load(state[0, y])
                # state_tile_imag = nl.load(state[1, y])
                state_tile_real = nl.ndarray((unitary_size, batch_size), dtype=state.dtype, buffer=nl.sbuf)
                state_tile_imag = nl.ndarray((unitary_size, batch_size), dtype=state.dtype, buffer=nl.sbuf)
                for i in nl.affine_range(batch_size):
                    # idx = nl.copy(y[:, i])
                    state_tile_real[:, i] = nl.load(state[0, y[:, i].as_tile()])
                    state_tile_imag[:, i] = nl.load(state[1, y[:, i].as_tile()])

                # See https://www.cs.utexas.edu/~flame/pubs/flawn81.pdf, pg10, eq4
                W_real = nl.matmul(U_tile_real, state_tile_real)
                W_imag = nl.matmul(U_tile_imag, state_tile_imag)
                SASB = nl.matmul(nl.add(U_tile_real, U_tile_imag), nl.add(state_tile_real, state_tile_imag))
                new_state_tile_real = nl.subtract(W_real, W_imag)
                new_state_tile_imag = nl.subtract(nl.subtract(SASB, W_real), W_imag)

                # nl.store(state[0, y], value=new_state_tile_real)
                # nl.store(state[1, y], value=new_state_tile_imag)
                for i in nl.affine_range(batch_size):
                    # idx = nl.copy(y[:, i])
                    nl.store(state[0, y[:, i].as_tile()], value=new_state_tile_real[:, i])
                    nl.store(state[1, y[:, i].as_tile()], value=new_state_tile_imag[:, i])

        return state



    # Helper function in CPU, generates the permutation info required by the kernel.
    # Args:
    #  - n: number of qubits
    #  - gate_idcs: qubit indices that the gate acts on (qiskit format)
    # Returns:
    #    List of src/dst qubit index pairs, shape = (m, 4).
    #    For efficiency, each row has the following numbers:
    #     - src
    #     - dst
    #     - 2^src
    #     - 2^dst
    @staticmethod
    def get_perm(n: int, gate_idcs: list[int]) -> list[np.ndarray]:
        perm = list(range(n))
        locs = list(range(n))
        for i, idx in enumerate(gate_idcs):
            if perm[i] != idx:
                # Swap idx elem in perm to i in perm
                loc_idx = locs[idx]
                perm[i], perm[loc_idx] = perm[loc_idx], perm[i]
                locs[perm[i]] = i
                locs[perm[loc_idx]] = loc_idx

        movements = []
        for i, idx in enumerate(perm):
            # if i != idx:
            #     movements.append([i, idx])
            movements.append([i, idx])
        # if len(movements) == 0:
        #     return np.array([[0, 0, 1, 1]], dtype=np.uint32)

        movements_arr = np.array(movements, dtype=np.uint32)
        return np.concatenate((movements_arr, np.left_shift(1, movements_arr)),
                              axis=1)

    # Basic gate consolidation algorithm.
    # We just keep combining gates until we go over seven qubits
    def consolidate(self):
        gates = []

        curr_gate = qiskit.QuantumCircuit(self.gate_size)
        curr_idcs = []
        curr_idcs_inv = {}

        # Consolidate the gate described by (U, idcs) into curr_gate
        def add_to_gate(U, idcs):
            for idx in idcs:
                if idx not in curr_idcs:
                    curr_idcs_inv[idx] = len(curr_idcs)
                    curr_idcs.append(idx)
            assert len(curr_idcs) <= self.gate_size
            curr_gate.append(UnitaryGate(U), [curr_idcs_inv[i] for i in idcs])

        # Add curr_gate to the output list of gates, and reset it back to identity
        def flush_gate():
            nonlocal curr_idcs, curr_idcs_inv, curr_gate

            # Add qubits until the curr_gate size reaches self.gate_size
            for i in range(self.n):
                if len(curr_idcs) == self.gate_size:
                    break
                if i in curr_idcs:
                    continue
                curr_idcs.append(i)
            assert len(curr_idcs) == self.gate_size

            # Convert curr_gate (a qiskit circuit) to a unitary matrix
            gates.append((Operator(curr_gate).data, curr_idcs))

            curr_gate = qiskit.QuantumCircuit(self.gate_size)
            curr_idcs = []
            curr_idcs_inv = {}

        for U, idcs in self.gates:
            can_consolidate = len(set(idcs) - set(curr_idcs)) <= self.gate_size - len(curr_idcs)
            if not can_consolidate:
                flush_gate()
            add_to_gate(U, idcs)

        flush_gate()

        self.gates = gates
