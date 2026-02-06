import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def random_perm(n):
    return tuple(np.random.choice(range(n), size=n, replace=False))


def test(n):
    shape = tuple([2] * n)
    x = torch.rand(shape)

    start = time.time();
    y = x.permute(random_perm(n)).flatten()
    end = time.time();
    return end - start, x

ns = range(1, 30)
ts = []
for n in ns:
    t, _ = test(n)
    print(f'{n=}, {t=}')
    ts.append(t)

plt.plot(ns, ts)
plt.yscale('log')
plt.grid(True, which='both')
plt.show()
