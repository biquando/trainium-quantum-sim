import numpy as np
import matplotlib.pyplot as plt

import _data as d


plt.plot(d.ns_trainium, d.trainium_full_without_compile_src, label='All-Trainium')
plt.plot(d.ns, d.custom_bora_full_src, label='CPU-only (Custom)')
plt.plot(d.ns, d.hybrid_full_src, label='Hybrid')
plt.plot(d.ns, d.qiskit_bora_default, label='CPU-only (Qiskit)')
# plt.plot(d.ns, d.qiskit_trn1_default, label='CPU-only (Qiskit) on trn1')
# plt.plot(d.ns, d.custom_trn1_full_src, label='CPU-only (Custom) on trn1')
plt.xlabel('Number of qubits')
plt.ylabel('Execution time per source gate (sec)')
plt.yscale('log')
# plt.axis((15, 28, 1e-5, 1e2))
plt.legend()
plt.savefig('per-gate.png', dpi=300)
plt.clf()


plt.plot(d.ns, d.custom_bora_full_fused, label='CPU-only (Custom)')
plt.plot(d.ns, d.hybrid_full_fused, label='Hybrid')
plt.plot(d.ns, d.qiskit_bora_consolidate_fused, label='CPU-only (Qiskit)')
# plt.plot(d.ns, d.qiskit_trn1_consolidate_fused, label='CPU-only (Qiskit) on trn1')
# plt.plot(d.ns, d.custom_trn1_full_fused, label='CPU-only (Custom) on trn1')
plt.xlabel('Number of qubits')
plt.ylabel('Execution time per 7-qubit gate (sec)')
plt.yscale('log')
plt.legend()
plt.savefig('consolidated.png', dpi=300)
plt.clf()



# plt.plot(d.ns, d.hybrid_full_fused, label='Full implementation')
# plt.plot(d.ns, d.hybrid_noperm_fused, label='No permute')
# plt.plot(d.ns, d.hybrid_nocpu_fused, label='No permute or memory transfer')
# plt.xlabel('Number of qubits')
# plt.ylabel('Execution time per 7-qubit gate (sec)')
# plt.yscale('log')
# plt.legend()
# plt.savefig('mem-overhead-abs.png', dpi=300)
# plt.clf()


xs = []
ys = []
zs = []
for xyz, yz, z in zip(d.hybrid_full_fused, d.hybrid_noperm_fused, d.hybrid_nocpu_fused):
    xs.append(xyz - yz)
    ys.append(yz - z)
    zs.append(z)
plt.plot(d.ns, ys, label='Transfering')
plt.plot(d.ns, xs, label='Permuting')
plt.plot(d.ns, zs, label='Block-diagonal matmul')
plt.xlabel('Number of qubits')
plt.ylabel('Execution time per 7-qubit gate (sec)')
plt.yscale('log')
plt.legend()
plt.savefig('mem-overhead-breakdown.png', dpi=300)
plt.clf()


trainium_full_compile_time = [x-y for x,y in zip(d.trainium_full_with_compile_fused, d.trainium_full_without_compile_fused)]
plt.plot(d.ns_trainium, trainium_full_compile_time, label='Compile time')
plt.plot(d.ns_trainium, d.trainium_full_without_compile_fused, label='Execution time')
plt.xlabel('Number of qubits')
plt.ylabel('Execution time per 7-qubit gate (sec)')
plt.yscale('log')
plt.legend()
plt.savefig('trainium-compile-time.png', dpi=300)
plt.clf()


plt.plot(d.ns_consolidation, d.qft_consolidation_ratios, label='QFT')
plt.plot(d.ns_consolidation, d.random_consolidation_ratios, label='Random 2-qubit gates')
plt.xlabel('Number of qubits')
plt.ylabel('Ratio of source-level gates to 7-qubit gates')
plt.axis((15, 41, 0, 30))
plt.legend()
plt.savefig('consolidation-ratios.png', dpi=300)
plt.clf()


plt.plot(d.ns, d.qiskit_bora_serial, label='CPU-only (Qiskit), serial')
plt.plot(d.ns, d.hybrid_full_src, label='Hybrid')
plt.plot(d.ns, d.qiskit_bora_default, label='CPU-only (Qiskit), parallel')
plt.xlabel('Number of qubits')
plt.ylabel('Execution time per source gate (sec)')
plt.yscale('log')
plt.legend()
plt.savefig('serial.png', dpi=300)
plt.clf()
