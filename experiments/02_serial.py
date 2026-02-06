from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

n = 10
NGATES = 1000

def get_sizes():

    np.random.seed(0)
    unitary_mats = scipy.stats.unitary_group(dim=128, seed=0)

    # Generate some random gates
    unitaries = []
    qubit_indices = []
    for _ in range(NGATES):
        U = unitary_mats.rvs().astype(np.complex128)
        unitaries.append(U)

        idcs = np.random.choice(range(n), size=7, replace=False)
        qubit_indices.append(idcs)


    ngates_vals = range(10, NGATES + 10, 10)
    sizes = []
    for ngates in ngates_vals:
        qc = QuantumCircuit(n)
        for U, idcs in zip(unitaries[:ngates], qubit_indices[:ngates]):
            qc.append(UnitaryGate(U), list(idcs))

        with open('qc.qpy', 'wb') as fd:
            qpy.dump(qc, fd)
        # with open('qc.qpy', 'rb') as fd:
        #     new_qc = qpy.load(fd)[0]

        size = os.path.getsize('qc.qpy')
        print(f'{ngates=}\t{size=}')
        sizes.append(size)

    print(list(ngates_vals))
    print(sizes)

x = list(range(10, NGATES + 10, 10))
y = [2623799, 5247399, 7870999, 10494599, 13118199, 15741799, 18365399, 20988999, 23612599, 26236199, 28859799, 31483399, 34106999, 36730599, 39354199, 41977799, 44601399, 47224999, 49848599, 52472199, 55095799, 57719399, 60342999, 62966599, 65590199, 68213799, 70837399, 73460999, 76084599, 78708199, 81331799, 83955399, 86578999, 89202599, 91826199, 94449799, 97073399, 99696999, 102320599 , 104944199, 107567799, 110191399, 112814999, 115438599, 118062199, 120685799, 123309399, 125932999, 128556599, 131180199, 133803799, 136427399, 139050999, 141674599, 144298199, 146921799, 149545399, 152168999, 154792599, 157416199, 160039799, 162663399, 165286999, 167910599, 170534199, 173157799, 175781399, 178404999, 181028599, 183652199, 186275799, 188899399, 191522999, 194146599, 196770199, 199393799, 202017399, 204640999, 207264599, 209888199, 212511799, 215135399, 217758999, 220382599, 223006199, 225629799, 228253399, 230876999, 233500599, 236124199, 238747799, 241371399, 243994999, 246618599, 249242199, 251865799, 254489399, 257112999, 259736599, 262360199]

y2 = [x_i * (128/8 * 128 * 128 + 32/8 * 7) for x_i in x]

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
