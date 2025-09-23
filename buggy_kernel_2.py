import numpy as np

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# This kernel works normally, but it does not compile if we use the
# @nki.compiler.skip_middle_end_transformations flag.

# @nki.compiler.skip_middle_end_transformations
@nki.jit(mode='baremetal')
def buggy_kernel(U, x):
    U_tile = nl.load(U)
    x_tile = nl.load(x)

    res_tile = nl.matmul(U_tile, x_tile)

    res = nl.ndarray(x.shape, dtype=res_tile.dtype, buffer=nl.shared_hbm)
    nl.store(res, res_tile)
    return res


def main():
    dim = 128
    U = np.random.random((dim, dim)).astype(np.float32)
    x = np.random.random((dim, 1)).astype(np.float32)
    print(buggy_kernel(U, x))


if __name__ == '__main__':
    main()
