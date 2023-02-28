# Project Name: Quantum Machine Learning with Quantum Topological Data Analysis

## Team Name: AnK
### Team Members: Ankit Khandelwal

#### Files:
The project is contained in Jupyter notebooks that are meant to be taken together as the project.
Here, I will provide some description of the project files.
***
* intro.??? contains the introduction to Topological Data Analysis (TDA), Betti Numbers and Quantum Topological Data Analysis (QTDA)
* mnist_qtda_qml.ipynb shows different approaches to use Betti numbers for classifying 0 and 1 digits. The simulations were performed on the Run:ai server using the provided NVIDIA GPU. A hybrid quantum classical machine learning model performs best. (See get_system_ready.txt)
* mnist_edge_length.ipynb uses variable edge lengths to find the optimal edge length for simplicial complex creation to use for classification.
* torus_vs_swiss_roll.ipynb highlights the benefit of TDA and QTDA by classifying the two different shapes embeded in a 10 dimensional space. It also shows that the method can handle noise in the data.
* torus_swiss_roll_aws.ipynb shows the effect of increasing the number of precision qubits in the algorithm. These simulations were performed on the SV1 AWS simulator and the local simulator provided by `qiskit_braket_provider`.
* quantum_betti_calc.ipynb ???
* qtda_decompose.py shows how pauli decomposition is used to obtain the circuit for the unitaries. Reducing the qubit count and depth is one of the planned future tasks. This script was run on cuquantum-appliance:22.11 utilising NVIDIA GPU. The number of precision qubits can be changed from line 53 in the file. Depending on the device, GPU usage can be changed from line 69.

***
* images directory contains some images that are used in the notebooks
* utils.py contains some utility functions
* classical_betti_calc.py contains helper functions to classically calculate Betti numbers
* get_system_ready.txt contains the commands used to get the Run:ai server setup to run the simulations.
* requirements.txt contains the basic project requirements. Using the GPU and/or AWS will require further setup dependent on the individual device.
***

## References:
???

Lloyd, S., Garnerone, S. & Zanardi, P. Quantum algorithms for topological and geometric analysis of data. Nat Commun 7, 10138 (2016). https://doi.org/10.1038/ncomms10138
