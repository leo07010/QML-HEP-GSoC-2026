# QML-HEP GSoC 2026 — Evaluation Tasks

This repository contains my solutions for the [QML-HEP Google Summer of Code 2026](https://ml4sci.org/) evaluation tasks.

**Applicant:** Leo
**Projects of Interest:**
- Q-MAML — Quantum Model-Agnostic Meta-Learning for Variational Quantum Algorithms for HEP Analysis at the LHC
- Quantum Reinforcement Learning for High Energy Physics
- Automated Scientific Discovery of Quantum Machine Learning Architectures

## Completed Tasks

| Task | Notebook | Description |
|------|----------|-------------|
| **I** | [`Task_I_Quantum_Computing.ipynb`](Task_I_Quantum_Computing.ipynb) | Quantum circuits with Cirq & PennyLane |
| **II** | [`Task_II_Classical_GNN.ipynb`](Task_II_Classical_GNN.ipynb) | GNN-based Quark/Gluon jet classification |
| **III** | [`Task_III_Open_Task.ipynb`](Task_III_Open_Task.ipynb) | Reflections on QML — honest observations and a path forward |
| **XI** | [`Task_XI_MLP_PQC.ipynb`](Task_XI_MLP_PQC.ipynb) | MLP + Parameterized Quantum Circuit embedding |

---

## Task I: Quantum Computing

**Part 1 — Cirq:** 5-qubit circuit with Hadamard, CNOT chain, SWAP, and Rx(π/2) gates.

**Part 2 — PennyLane:** SWAP test circuit comparing two 2-qubit states using an ancilla qubit with CSWAP gates. Measures the fidelity |⟨ψ|φ⟩|² = 0.5 between the states.

## Task II: Classical GNN

Two graph-based architectures for Quark/Gluon jet classification using the [ParticleNet dataset](https://zenodo.org/records/3164691):

| Model | Parameters | Test Accuracy | Test AUC |
|-------|-----------|---------------|----------|
| **EdgeConv** | 67K | 0.789 | **0.860** |
| **GATv2** | 219K | 0.787 | 0.857 |

**Graph construction:** Each particle becomes a node; edges are built using k=7 nearest neighbors in (η, φ) space, which respects the angular geometry of particle detectors.

## Task III: Open Task

Reflections on QML: argues that most current QML is better described as "quantum-inspired ML" since classical simulators handle it fine. Discusses the software ecosystem evolution (Qiskit → PennyLane → CUDA-Q) and advocates for hybrid approaches where classical ML assists quantum computing (e.g., Transformer-based circuit generation) and quantum components enhance classical models (e.g., QCBM-augmented VAEs for molecular design).

## Task XI: MLP + PQC Embedding

Hybrid classical-quantum model that maps normally distributed input to quantum state expectations:

- **MLP:** 8 → 32 → 32 → 30 (3 Linear layers with Tanh activation)
- **PQC:** 5 qubits, 2 layers of (RX, RY, RZ) + circular CNOT entangling
- **Loss:** MSE between predicted and target ⟨Z⟩ expectation values
- **Final Test MSE:** 0.012

---

## Environment

- **Framework:** PyTorch 2.6, PennyLane 0.38, Cirq 1.3, PyG 2.6
- **Hardware:** NVIDIA GPUs via SLURM (TWCC HPC)
- **Python:** 3.9

## Repository Structure

```
├── README.md
├── Task_I_Quantum_Computing.ipynb    # Quantum circuits (Cirq + PennyLane)
├── Task_II_Classical_GNN.ipynb       # GNN jet classification
├── Task_III_Open_Task.ipynb          # Open task — QML reflections
├── Task_XI_MLP_PQC.ipynb             # Hybrid MLP + PQC embedding
├── task1/                            # Task I source code & outputs
├── task2/                            # Task II source code, data & models
│   └── data/QG_jets.npz
└── task11/                           # Task XI source code & outputs
```
