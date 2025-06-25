import numpy as np

from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit(mode='baremetal')
def buggy_kernel(x):
    xtile = nl.load(x)
    ytile = nl.copy(xtile)

    # This can be any operation that modifies xtile.
    #
    # In theory, this operation should not affect the result. This is because
    # we are returning ytile, which is a *copy* of xtile.
    #
    # In simulation, we get [1], as expected.
    # On baremetal, we get [2], which means that xtile and ytile point to the
    # same memory, even though ytile should be a copy.
    xtile[:] = nl.add(xtile, xtile)

    y = nl.ndarray(ytile.shape, dtype=ytile.dtype, buffer=nl.shared_hbm)
    nl.store(y, value=ytile)
    return y

def main():
    x = np.array([[1]], dtype=np.float16)
    print(buggy_kernel(x))


if __name__ == '__main__':
    main()
