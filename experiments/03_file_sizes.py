import sys
import time
import numpy as np
from qiskit import QuantumCircuit, qasm2, qasm3, qpy
from qiskit.quantum_info import random_unitary

n = 20
gate_size = 7
ngates = 20

np.random.seed(0)

start = time.time()
qc = QuantumCircuit(n)
for _ in range(ngates):
    U = random_unitary(2 ** gate_size)
    idcs = np.random.choice(range(n), size=gate_size, replace=False)
    qc.append(U, list(idcs))
end = time.time()
print('Generation time:', end - start)

start = time.time()
match sys.argv[1]:
    case 'qasm2':
        with open('qc.qasm2', 'w') as f:
            qasm2.dump(qc, f)
    case 'qasm3':
        with open('qc.qasm3', 'w') as f:
            qasm3.dump(qc, f)
    case _:
        with open('qc.qpy', 'wb') as f:
            qpy.dump(qc, f)
end = time.time()
print(f'{sys.argv[1]} time:', end - start)
