import numpy as np
import scipy
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector


n = 10                          # number of qubits
N = 2 ** n                      # length of state vector
gate_size = 7                   # number of qubits per gate (must be at most 7)
unitary_size = 2 ** gate_size   # size of each gate's unitary matrix
ntiles = 2 ** (n - gate_size)   # number of tiles that we must multiply

DTYPE = np.float32


# Helper function in CPU, generates the permutation info required by the kernel.
# This code is not great, it's just the first algorithm I came up with.
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
def get_perm(n, gate_idcs):
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
        if i != idx:
            movements.append([i, idx])
    if len(movements) == 0:
        return np.array([[0, 0, 1, 1]], dtype=np.uint32)

    movements_arr = np.array(movements, dtype=np.uint32)
    return np.concatenate((movements_arr, np.left_shift(1, movements_arr)),
                          axis=1)


# initial_state: numpy array state vector with 2^n elements
# gates: list of tuples
#            - first element of tuple is the real component of the matrix
#            - second element of tuple is the imaginary component of the matrix
#            - third element of tuple is idx permutation info
#              perm: list of src/dst qubit index pairs, shape = (n, 4)
#                    for efficiency, each row has the following numbers:
#                     - src
#                     - dst
#                     - 2^src
#                     - 2^dst
# returns: final state vector
@nki.jit(mode='simulation')
def run_circuit(gates):
    # Initialize state vector to |0>
    state = nl.ndarray((2, N, 1), dtype=DTYPE, buffer=nl.shared_hbm)
    for tile_idx in nl.affine_range(ntiles):
        nl.store(dst=state[0, tile_idx * unitary_size : (tile_idx+1) * unitary_size, 0], value=0.0)
        nl.store(dst=state[1, tile_idx * unitary_size : (tile_idx+1) * unitary_size, 0], value=0.0)
    nl.store(dst=state[0, 0, 0], value=1.0)

    # This just stores [0, 1, 2, ..., 127], for use in indexing
    tile_idcs_base_range = nl.arange(unitary_size)[:, None]
    tile_idcs_base = nisa.iota(tile_idcs_base_range, dtype=np.uint32)

    for gate in gates:
        assert type(gate) == tuple and len(gate) == 3
        assert gate[0].shape == (unitary_size, unitary_size)
        assert gate[1].shape == (unitary_size, unitary_size)
        assert gate[2].shape[1] == 4

        # This is the 128x128 matrix corresponding the current gate
        U_tile_real = nl.load(gate[0])
        U_tile_imag = nl.load(gate[1])

        # We apply the matrix to each chunk of the state vector
        for tile_idx in nl.affine_range(ntiles): # allows for parallel computation
            offset = tile_idx * unitary_size
            tile_idcs = nl.add(tile_idcs_base, offset, dtype=np.uint32)

            # Convert the qubit permutation to a statevector permutation using bit operations
            y = nl.copy(tile_idcs) # permuted tile idcs
            for i in nl.sequential_range(gate[2].shape[0]):
                a  = nl.load(gate[2][i][0])
                b  = nl.load(gate[2][i][1])
                ma = nl.load(gate[2][i][2]) # 2^a
                mb = nl.load(gate[2][i][3]) # 2^b

                y[:] = nl.bitwise_and(y, nl.invert(mb))
                y[:] = nl.bitwise_or(y, nl.left_shift(nl.bitwise_and(tile_idcs, ma), b - a))
                y[:] = nl.bitwise_or(y, nl.right_shift(nl.bitwise_and(tile_idcs, ma), a - b))


            # *** Load/Multiply/Store (3M) ***

            state_tile_real = nl.load(state[0, y])
            state_tile_imag = nl.load(state[1, y])

            # See https://www.cs.utexas.edu/~flame/pubs/flawn81.pdf, pg10, eq4
            W_real = nl.matmul(U_tile_real, state_tile_real)
            W_imag = nl.matmul(U_tile_imag, state_tile_imag)
            SASB = nl.matmul(nl.add(U_tile_real, U_tile_imag), nl.add(state_tile_real, state_tile_imag))
            new_state_tile_real = nl.subtract(W_real, W_imag)
            new_state_tile_imag = nl.subtract(nl.subtract(SASB, W_real), W_imag)

            nl.store(state[0, y], value=new_state_tile_real)
            nl.store(state[1, y], value=new_state_tile_imag)

    return state


def main():
    np.random.seed(0)
    unitary_mats = scipy.stats.unitary_group(dim=unitary_size, seed=0)

    # Generate some random gates
    NGATES = 20
    unitaries = []
    qubit_indices = []
    for _ in range(NGATES):
        U = unitary_mats.rvs().astype(np.complex128)
        unitaries.append(U)

        idcs = np.random.choice(range(n), size=gate_size, replace=False)
        qubit_indices.append(idcs)

    # Simulate circuit in kernel
    # Before passing the gates into the kernel, we have to:
    #  1. Separate complex matrices into two real matrices
    #  2. Convert the matrices to fp16 or fp32
    #  3. Precompute some qubit permutation information
    kernel_gates = [
        (U.real.astype(DTYPE), U.imag.astype(DTYPE), get_perm(n, idcs))
        for U, idcs in zip(unitaries, qubit_indices)
    ]
    kernel_state = run_circuit(kernel_gates)
    kernel_state = kernel_state[0].astype(np.complex128) \
                 + kernel_state[1].astype(np.complex128) * 1.0j

    # Simulate circuit in Qiskit
    qc = QuantumCircuit(n)
    for U, idcs in zip(unitaries, qubit_indices):
        qc.append(UnitaryGate(U), list(idcs))
    qiskit_state = Statevector.from_instruction(qc).data.astype(np.complex128)

    # Calculate the Euclidean distance between the two state vectors
    diff = kernel_state - qiskit_state[:, None]
    print('Distance between state vectors:', np.linalg.norm(diff))


if __name__ == '__main__':
    main()
