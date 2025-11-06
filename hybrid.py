import numpy as np
import time
import os

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.typing as nt

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm

import qiskit
from qiskit.circuit.library import UnitaryGate, QFTGate, QFT
from qiskit.quantum_info import Operator

import circuit


UNITARY_SIZE = 128
BATCH_SIZE = 128



@nki.jit
def kernel(initial_state, U):
    # load unitary
    state = nl.ndarray(initial_state.shape, dtype=initial_state.dtype, buffer=nl.shared_hbm)
    U_tile_real = nl.load(U[0])
    U_tile_imag = nl.load(U[1])
    assert U_tile_real.shape == (UNITARY_SIZE, UNITARY_SIZE)
    assert U_tile_imag.shape == (UNITARY_SIZE, UNITARY_SIZE)

    for i in nl.affine_range(state.shape[1] // UNITARY_SIZE):
        idx_start = i * UNITARY_SIZE
        idx_end = (i+1) * UNITARY_SIZE

        # load
        # i_p, i_f = nl.mgrid[idx_start:idx_end, 0:UNITARY_SIZE]
        # state_tile_real = nl.load(state[0, i_p, i_f])
        # state_tile_imag = nl.load(state[1, i_p, i_f])
        state_tile_real = nl.ndarray((UNITARY_SIZE, BATCH_SIZE), dtype=state.dtype)
        state_tile_imag = nl.ndarray((UNITARY_SIZE, BATCH_SIZE), dtype=state.dtype)
        state_tile_real[...] = nl.load_transpose2d(initial_state[0, idx_start:idx_end, 0:UNITARY_SIZE])
        state_tile_imag[...] = nl.load_transpose2d(initial_state[1, idx_start:idx_end, 0:UNITARY_SIZE])
        # state_tile_real = nl.transpose(state_tile_real)
        # state_tile_imag = nl.transpose(state_tile_imag)


        # multiply
        # We have U and S^T. We want U @ S.
        # S @ U = (U @ S)^T
        # nisa.nc_matmul expects a transposed lhs, so this would require only one transpose
        # state_tile_real = nl.transpose(nisa.nc_matmul(state_tile_real, U_tile_real))
        # state_tile_imag = nl.transpose(nisa.nc_matmul(state_tile_imag, U_tile_imag))

        # W_real = nl.matmul(U_tile_real, state_tile_real)
        # W_imag = nl.matmul(U_tile_imag, state_tile_imag)
        # S_A = nl.add(U_tile_real, U_tile_imag)
        # S_B = nl.add(state_tile_real, state_tile_imag)
        # state_tile_real = nl.subtract(W_real, W_imag)
        # state_tile_imag = nl.subtract(nl.subtract(nl.matmul(S_A, S_B), W_real), W_imag)
        W_real = nl.matmul(U_tile_real, state_tile_real)
        W_imag = nl.matmul(U_tile_imag, state_tile_imag)
        SASB = nl.matmul(nl.add(U_tile_real, U_tile_imag), nl.add(state_tile_real, state_tile_imag))
        state_tile_real = nl.transpose(nl.subtract(W_real, W_imag))
        state_tile_imag = nl.transpose(nl.subtract(nl.subtract(SASB, W_real), W_imag))


        # store
        # nl.store(state[0, i_p, i_f], value=state_tile_imag)
        # nl.store(state[1, i_p, i_f], value=state_tile_real)
        nl.store(state[0, idx_start:idx_end, 0:UNITARY_SIZE], value=state_tile_imag)
        nl.store(state[1, idx_start:idx_end, 0:UNITARY_SIZE], value=state_tile_real)

    # nl.store(state[0, 0:128, 0:128], U_tile_real)
    # nl.store(state[1, 0:128, 0:128], U_tile_imag)
    return state


def run_circuit(state, gates, n, exclude=()):
    '''
    Args:
        state: initial state vector with shape (2, N)
        gates: list of 2-tuples, with unitaries and qubit indices
    '''
    for i, (U, idcs) in enumerate(gates):
        xm.mark_step()
        if 'print' not in exclude:
            print(f'gate {i+1} of {len(gates)}')
        if i == min(len(gates), 1):
            start = time.time() # we don't time the first gate because the time is spent compiling

        if 'permute' not in exclude:
            permutation = idcs + [idx for idx in range(n) if idx not in idcs]
            permutation = [0] + [idx+1 for idx in permutation]  # because the first axis is real/complex

            inv_permutation = [0] * (n+1)
            for i, dim in enumerate(permutation):
                inv_permutation[dim] = i

        if 'permute' not in exclude:
            state = state.reshape([2] * (n+1))
            state = torch.permute(state, permutation)
        state = state.reshape([2, -1, UNITARY_SIZE])
        state = state.to('xla')

        state = kernel(state, U.to('xla'))

        if 'cpu' not in exclude:
            state = state.to('cpu')
        state = state.reshape([2] * (n+1))
        if 'permute' not in exclude:
            state = torch.permute(state, inv_permutation)

    xm.mark_step()
    end = time.time()
    # print(f'time: {(end-start) * len(gates) / (len(gates)-1)}')
    exec_time = (end-start) * len(gates) / max(len(gates)-1, 1)

    return exec_time, state.reshape([2, -1])


class Circuit(nn.Module):
    def __init__(self, gates, n, exclude=()):
        self.gates = gates
        self.n = n
        self.exclude = exclude
        super().__init__()
    def forward(self,state):
        return run_circuit(state, self.gates, self.n, self.exclude)


dtype = torch.float32
ns = range(16, 28)
gate_times = []
src_gate_times = []
for n in ns:
    # n = 25

    # orig_qc = qiskit.QuantumCircuit(n)
    # orig_qc.h(0)
    # for i in range(1, n):
    #     orig_qc.cx(0, i)
    # orig_qc.append(QFTGate(n), range(n))
    orig_qc = qiskit.QuantumCircuit(n)
    orig_qc.append(QFTGate(n), range(n))
    orig_qc = qiskit.transpile(orig_qc, basis_gates=['u', 'cx'])
    n_src_gates = len(orig_qc.data)

    qc = circuit.QuantumCircuit.from_qiskit(orig_qc)
    qc.consolidate()
    n_gates = len(qc.gates)
    for i, gate in enumerate(qc.gates):
        torch_U = torch.from_numpy(gate[0])
        qc.gates[i] = (torch.stack((torch.real(torch_U).type(dtype),
                                    torch.imag(torch_U).type(dtype))),
                       gate[1])


    state = torch.zeros([2, 2**n], dtype=dtype)
    state[0, 0] = 1.0


    # model = Circuit(qc.gates, n, exclude={})
    # model = Circuit(qc.gates, n, exclude={'permute'})
    model = Circuit(qc.gates, n, exclude={'cpu', 'permute'})
    with torch.no_grad():
        exec_time, state = model(state)
    # exec_time, state = run_circuit(state, qc.gates, n, exclude={'print'})
    # state = run_circuit(state, qc.gates, n, exclude=('cpu', 'permute'))
    # print(state.shape)
    # print(state)
    state.to('cpu')
    src_gate_times.append(exec_time / n_src_gates)
    gate_times.append(exec_time / n_gates)

for n, src_gate_time, gate_time in zip(ns, src_gate_times, gate_times):
    print(f'{n=}, {src_gate_time=}, {gate_time=}')
