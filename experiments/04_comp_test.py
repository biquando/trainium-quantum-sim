import time
from scipy.stats import unitary_group, ortho_group
import numpy as np

from qiskit import QuantumCircuit

n = 20
N = 2 ** n
gate_size = 7
tile_size = 2 ** gate_size
NGATES = 20

unitary_mats = unitary_group(dim=tile_size, seed=0)

state = np.zeros((2, N, 1))
state[0, 0, 0] = 1.0
print(state.shape)

unitaries = []
# qubit_indices = []
for _ in range(NGATES):
    U = unitary_mats.rvs().astype(np.complex128)
    unitaries.append([U.real, U.imag])

    # idcs = np.random.choice(range(n), size=gate_size, replace=False)
    # qubit_indices.append(idcs)

start = time.time()
for unitary in unitaries:
    state = np.random.permutation(state)
    for tile in range(N // tile_size):
        offset = tile * tile_size
        # # See https://www.cs.utexas.edu/~flame/pubs/flawn81.pdf, pg10, eq4
        W_real = unitary[0] @ state[0, offset:offset+tile_size, 0]
        W_imag = unitary[1] @ state[1, offset:offset+tile_size, 0]
        SASB = (unitary[0] + unitary[1]) @ (state[0, offset:offset+tile_size, 0] + state[1, offset:offset+tile_size, 0])
        state[0, offset:offset+tile_size, 0] = W_real - W_imag
        state[1, offset:offset+tile_size, 0] = SASB - W_real - W_imag

end = time.time()

print(state)
print(end - start)
