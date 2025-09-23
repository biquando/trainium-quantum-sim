import torch
from torch import nn
import torch_xla.core.xla_model as xm

import numpy as np

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa


# This example contains a kernel that accepts a 128 by 128 matrix `U`,
# a 128 by 1 vector `x`, and a 128 by 1 matrix `idcs`. The kernel
# loads an arbitrary permutation of x (determined by idcs) into a
# 128 by 1 tile. Then we do a matrix multiply of U and x, and finally
# store the tile back into a 128 by 1 vector using the same
# permutation.

# This kernel successfully compiles normally, but it does not work
# with the @nki.compiler.skip_middle_end_transformations flag.

# The error message:
# 2025-09-11 12:23:08.000052:  9058  ERROR ||NEURON_CC_WRAPPER||: Failed compilation w
# ith ['neuronx-cc', 'compile', '--framework=XLA', '/tmp/ubuntu/neuroncc_compile_workd
# ir/fd7b0c1a-3b0d-4073-a7c0-b2011ed56f16/model.MODULE_12628662556016653404+e30acd3a.h
# lo_module.pb', '--output', '/tmp/ubuntu/neuroncc_compile_workdir/fd7b0c1a-3b0d-4073-
# a7c0-b2011ed56f16/model.MODULE_12628662556016653404+e30acd3a.neff', '--target=trn1',
#  '--verbose=35']: [DVR002]  Internal transformation failed - Please open a support t
# icket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new. You may also be ab
# le to obtain more information using the 'XLA_IR_DEBUG' and 'XLA_HLO_DEBUG' environme
# nt variables.


@nki.compiler.skip_middle_end_transformations
@nki.jit
def kernel(U, x, idcs):
    U_tile = nl.load(U)
    x_tile = nl.load(x)

    res_tile = nl.matmul(U_tile, x_tile)

    res = nl.ndarray(x.shape, dtype=res_tile.dtype, buffer=nl.shared_hbm)
    nl.store(res, res_tile)
    return res

class QC(nn.Module):
    def __init__(self, U, x, idcs):
        super(QC, self).__init__()
        self.register_buffer('U', torch.as_tensor(U, dtype=torch.float32))
        self.register_buffer('x', torch.as_tensor(x, dtype=torch.float32))
        self.register_buffer('idcs', torch.as_tensor(idcs, dtype=torch.uint32))
        assert self.U.shape == (128, 128)
        assert self.x.shape == (128, 1)
        assert self.idcs.shape == (128, 1)

    def forward(self):
        return kernel(self.U, self.x, self.idcs)


def main():
    device = xm.xla_device()
    device = 'xla'
    idcs = np.array(
        range(128),
        dtype=np.uint32
    ).reshape((1, 128)).T

    U = np.ones((128, 128), dtype=np.float32)
    x = np.ones((128, 1), dtype=np.float32)

    qc = QC(U, x, idcs).to(device)
    state = qc()
    print(state)

if __name__ == '__main__':
    main()
