from torch_circuit import QC
import circuit

import time

import scipy
import torch
import torch_xla.core.xla_model as xm

def main():
    device = 'xla'
    dtype = torch.float32
    qc = QC().to(device)

    unitaries = scipy.stats.ortho_group(dim=128, seed=0)
    U = unitaries.rvs()
    U = torch.from_numpy(U).to(device)
    x = torch.zeros((128, 1), dtype=dtype, device=device)
    x[0, 0] = 1

    start = time.time()
    res = qc(U, x)
    end = time.time()
    print(f'Model took {end-start} sec')

    res = res.to('cpu')
    print(res)

if __name__ == '__main__':
    with torch.no_grad():
        main()
