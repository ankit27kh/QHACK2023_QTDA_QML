import pennylane.numpy as np
from scipy.linalg import null_space, eigvals, expm
import pennylane as qml
from itertools import combinations
from classical_betti_calc import boundary, homology, betti
from utils import gershgorin, find_cliques

np.random.seed(0)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# Create random edges
edges = set(map("".join, combinations("01234567", 2)))  # Vertex labels
for _ in range(int(len(edges) * 0.3)):  # Remove 30% of edges
    edges.pop()

# Generate a simplicial complex
sxs = find_cliques(edges)
sxs1 = set()
for u in sxs[1:]:
    sxs1 = sxs1.union(u)
sc = list(sxs1)
# Randomly remove some simplicies to increase betti number values
# This might not technically fulfill all conditions for a simplicial complex but works as demonstration
np.random.shuffle(sc)
sc = sc[: int(len(sc) * 0.3)]
sc.extend(list(sxs[1]))
sc = list(set(sc))
sc.sort()

# Classically calculate betti numbers
bnd, simplicies = boundary(sc)  # Boundary operators
Homo = homology(bnd)
b = betti(Homo)  # Betti numbers
print("Betti Numbers:", b)

# Quantum calculation for k-th betti number
k = 0
# Min and max precision qubits to use for QPE
min_pre = 1
max_pre = 10

# Calculate the combinatorial laplacian and verify betti number classically
comb_lap = bnd[k].T @ bnd[k] + bnd[k + 1] @ bnd[k + 1].T
kernel_cp = null_space(comb_lap)
betti_k_dim = kernel_cp.shape[1]
# This is the value we want to estimate using quantum computing
betti_k_zero_eig = np.count_nonzero(np.absolute(eigvals(comb_lap)) < 10**-7)
print(f"Betti number {k}:", betti_k_dim, betti_k_zero_eig)
num = int(np.ceil(np.log2(comb_lap.shape[0])))

# Estimate max eigenvalue
max_eig = max(gershgorin(comb_lap), 1)

# Unitary generation for QPE
# Padding to make size a power of 2
# Scaling to get eigenvalues in [0,1)
H = np.identity(2**num) * max_eig / 2
H[: comb_lap.shape[0], : comb_lap.shape[0]] = comb_lap
H = H / max_eig * 6
U = expm(1j * H)

print(
    "Number of precision qubits",
    "|",
    "Estimated betti number with respective shots",
    "|",
    "Absolute error",
)

for pre in range(min_pre, max_pre + 1):
    dev = qml.device(
        "lightning.qubit",
        wires=2 * num + pre,
        shots=[10**1, 10**2, 10**3, 10**4, 10**5, 10**6],
    )

    @qml.qnode(dev)
    def circ():
        for i in range(num):
            qml.Hadamard(num + i)
        for i in range(num):
            qml.CNOT([num + i, i])
        qml.QuantumPhaseEstimation(
            U,
            target_wires=range(num, 2 * num),
            estimation_wires=range(2 * num, 2 * num + pre),
        )
        return qml.probs(range(2 * num, 2 * num + pre))

    probs = circ()
    print(
        pre,
        probs[:, 0] * 2**num,
        np.abs((betti_k_zero_eig - probs[:, 0] * 2**num)),
    )

print("Quantum Circuit")
print(
    qml.draw(
        circ,
        show_all_wires=True,
        show_matrices=False,
        max_length=10000,
        expansion_strategy="device",
    )()
)
