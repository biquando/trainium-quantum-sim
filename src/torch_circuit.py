import torch
from torch import nn
import circuit

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def kernel(U, x):
    Utile = nl.load(U)
    xtile = nl.load(x)
    restile = nl.matmul(U, x)
    res = nl.ndarray(restile.shape, dtype=restile.dtype, buffer=nl.shared_hbm)
    nl.store(res, value=restile)
    return res

class QC(nn.Module):
    def __init__(self, N, unitary_size, ntiles, batch_size, base):
        super(QC, self).__init__()
        # self.register_buffer('N', torch.tensor(N, dtype=torch.uint32))
        # self.register_buffer('unitary_size', torch.tensor(unitary_size, dtype=torch.uint32))
        # self.register_buffer('ntiles', torch.tensor(ntiles, dtype=torch.uint32))
        # self.register_buffer('batch_size', torch.tensor(batch_size, dtype=torch.uint32))
        # self.register_buffer('base', torch.as_tensor(base, dtype=torch.uint32))

        self.N = N
        self.unitary_size = unitary_size
        self.ntiles = ntiles
        self.batch_size = batch_size
        self.register_buffer('base', torch.as_tensor(base, dtype=torch.uint32))

        assert self.base.shape == (unitary_size, batch_size)

    def forward(self, unitaries, idcs):
        return circuit.QuantumCircuit.kernel(
            unitaries, idcs,
            self.N, self.unitary_size, self.ntiles, self.batch_size, self.base
        )
        # return circuit.QuantumCircuit.kernel(
        #     torch.zeros((128, 128), dtype=torch.float32, device='xla')
        # )
        # return kernel(unitaries, x)
