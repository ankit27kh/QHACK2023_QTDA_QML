import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.opflow import I, X, Z
from qiskit.circuit import Parameter
from qiskit.opflow import PauliTrotterEvolution

np.random.seed(0)
np.set_printoptions(precision=3)

"""
The considered simplicial complex is:
('0', '023', '1', '123', '2', '23', '3')
The combinatorial laplacian from this is:
[[ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  1., -1.],
 [ 0.,  0., -1.,  1.]]

And the considered hamiltonian is:
[[ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  3., -3.],
 [ 0.,  0., -3.,  3.]]
 
This hamiltonian can be pauli decomposed into:
h = -1.5 * (I ^ X) + -1.5 * (Z ^ I) + 1.5 * (I ^ I) + 1.5 * (Z ^ X)

We start from here.
We can get the circuit for the corresponding unitary from this decomposition.
"""

evo_time = Parameter("t")
h = -1.5 * (I ^ X) + -1.5 * (Z ^ I) + 1.5 * (I ^ I) + 1.5 * (Z ^ X)
exp_op = (evo_time * h).exp_i()
trotter_op = PauliTrotterEvolution().convert(exp_op)
U = trotter_op.bind_parameters({evo_time: -1})

print("The circuit for U is")
print(U.to_circuit().decompose().decompose().draw())
"""
global phase: 1.5
     ┌─────────┐┌────────────┐
q_0: ┤ Rx(3.0) ├┤1           ├
     ├─────────┤│  Rzx(-3.0) │
q_1: ┤ Rz(3.0) ├┤0           ├
     └─────────┘└────────────┘
"""


num = h.num_qubits  # Number of qubits for the hamiltonian

pre = 10  # Number of precision qubits
shots = 1000  # Number of shots


def qft_dagger(qc, n):
    """n-qubit QFTdagger the first n qubits in circ"""
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)


simulator = AerSimulator(
    method="statevector",
    device="GPU"
)


qc = QuantumCircuit(num + pre, pre)

qc.h(range(pre, num + pre))
for i in range(num):
    qc.cx(i + pre, i - num + pre)
for i in range(num):
    qc.reset(pre - 1 - i)
qc.barrier()
qc.h(range(pre))

for i in range(1, pre + 1):
    qc.append(
        (
            (trotter_op.bind_parameters({evo_time: -(2 ** (pre - i))})).to_circuit()
        ).control(1),
        [i - 1] + list(range(pre, num + pre)),
    )
qc.barrier()
qft_dagger(qc, pre)
qc.barrier()
qc.measure(range(pre), range(pre))
qc = transpile(qc, simulator)


result = simulator.run(qc, shots=shots).result()
counts = result.get_counts(qc)
print("The estimated Betti number is")
print((counts["0" * pre] / shots) * 2**num)
print("The actual Betti number is 3")
"""
Results:
Precision qubits    |    Estimated Betti number
1                   |       4
5                   |       3.14
10                  |       3.012
15                  |       3.004
"""