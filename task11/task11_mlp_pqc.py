"""
Task XI: MLP + Parameterized Quantum Circuit (PQC) Embedding

Pipeline:
  1. Generate normally distributed input data with target quantum expectation values
  2. MLP (3 Linear layers) maps input → PQC parameters
  3. PQC (5 qubits, 2 layers) prepares a quantum state from those parameters
  4. Measure expectation values of Pauli-Z on each qubit
  5. Train end-to-end with MSE Loss

Framework: PyTorch + PennyLane
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")  # PennyLane hybrid runs on CPU

# ============================================================
# 1. Quantum Circuit Setup (5 qubits, 2 ansatz layers)
# ============================================================
N_QUBITS = 5
N_LAYERS = 2
N_PQC_PARAMS = N_QUBITS * N_LAYERS * 3  # 3 rotations (RX, RY, RZ) per qubit per layer = 30

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(params):
    """
    Parameterized Quantum Circuit:
    - 2 layers of: single-qubit rotations (RX, RY, RZ) + entangling CNOT ring
    - Returns Pauli-Z expectation on each qubit
    """
    params = params.reshape(N_LAYERS, N_QUBITS, 3)

    for layer in range(N_LAYERS):
        # Single-qubit rotations
        for q in range(N_QUBITS):
            qml.RX(params[layer, q, 0], wires=q)
            qml.RY(params[layer, q, 1], wires=q)
            qml.RZ(params[layer, q, 2], wires=q)

        # Entangling layer: circular CNOT
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ============================================================
# 2. Data Generation
# ============================================================
INPUT_DIM = 8
N_TRAIN = 800
N_TEST = 200

# Fixed target mapping (shared between train and test)
np.random.seed(0)
W_target = np.random.randn(INPUT_DIM, N_QUBITS).astype(np.float64) * 0.5
b_target = np.random.randn(N_QUBITS).astype(np.float64) * 0.1
np.random.seed(42)


def generate_data(n_samples, input_dim=8):
    """
    Generate normally distributed input data and target expectation values.
    Targets are created by a fixed linear mapping from input to
    [-1, 1] range (valid range for Pauli-Z expectations).
    """
    X = np.random.randn(n_samples, input_dim).astype(np.float64)
    y = np.tanh(X @ W_target + b_target)  # tanh ensures [-1, 1]
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


X_train, y_train = generate_data(N_TRAIN, INPUT_DIM)
X_test, y_test = generate_data(N_TEST, INPUT_DIM)

print(f"Input dim: {INPUT_DIM}")
print(f"Qubits: {N_QUBITS}, PQC layers: {N_LAYERS}, PQC params: {N_PQC_PARAMS}")
print(f"Train: {N_TRAIN}, Test: {N_TEST}")
print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")


# ============================================================
# 3. Hybrid MLP + PQC Model
# ============================================================
class HybridMLPPQC(nn.Module):
    """
    Hybrid classical-quantum model:
      MLP (3 Linear layers) → PQC parameters → Quantum state → <Z> expectations

    The MLP learns to map classical input to the right PQC rotation angles
    so that the resulting quantum state's measurements match the targets.
    """
    def __init__(self, input_dim, n_pqc_params):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(32, 32, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(32, n_pqc_params, dtype=torch.float64),
        )

    def forward(self, x):
        # MLP estimates PQC parameters
        pqc_params = self.mlp(x)  # (batch, n_pqc_params)

        # Run each sample through PQC
        expectations = []
        for i in range(x.shape[0]):
            exp_vals = quantum_circuit(pqc_params[i])
            expectations.append(torch.stack(exp_vals))

        return torch.stack(expectations)  # (batch, n_qubits)


# ============================================================
# 4. Training
# ============================================================
model = HybridMLPPQC(INPUT_DIM, N_PQC_PARAMS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-3)
criterion = nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"  MLP: {INPUT_DIM}→64→32→{N_PQC_PARAMS}")
print(f"  PQC: {N_LAYERS} layers × {N_QUBITS} qubits × 3 rotations")

EPOCHS = 80
BATCH_SIZE = 64
train_losses = []
test_losses = []

print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE}")
print("-" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    # Mini-batch training
    indices = torch.randperm(N_TRAIN)
    for start in range(0, N_TRAIN, BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]
        x_batch = X_train[idx].to(DEVICE)
        y_batch = y_train[idx].to(DEVICE)

        optimizer.zero_grad()
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_loss / n_batches
    train_losses.append(avg_train_loss)

    # Evaluate on test set (small enough to do in one pass)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test.to(DEVICE))
        test_loss = criterion(test_pred, y_test.to(DEVICE)).item()
    test_losses.append(test_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS} | Train MSE: {avg_train_loss:.6f} | Test MSE: {test_loss:.6f}")


# ============================================================
# 5. Final Evaluation & Visualization
# ============================================================
model.eval()
with torch.no_grad():
    final_pred = model(X_test.to(DEVICE)).cpu().numpy()
    final_target = y_test.numpy()

final_mse = np.mean((final_pred - final_target) ** 2)
print(f"\nFinal Test MSE: {final_mse:.6f}")

# --- Plot 1: Training curve ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label="Train MSE")
axes[0].plot(test_losses, label="Test MSE")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Plot 2: Predicted vs Target per qubit ---
for q in range(N_QUBITS):
    axes[1].scatter(final_target[:, q], final_pred[:, q], alpha=0.4, s=10, label=f"Q{q}")
axes[1].plot([-1, 1], [-1, 1], "k--", alpha=0.5)
axes[1].set_xlabel("Target ⟨Z⟩")
axes[1].set_ylabel("Predicted ⟨Z⟩")
axes[1].set_title("Predicted vs Target (per qubit)")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-1.1, 1.1)
axes[1].set_ylim(-1.1, 1.1)

# --- Plot 3: Residual distribution ---
residuals = final_pred - final_target
axes[2].hist(residuals.flatten(), bins=40, edgecolor="black", alpha=0.7)
axes[2].axvline(0, color="red", linestyle="--")
axes[2].set_xlabel("Residual (Pred - Target)")
axes[2].set_ylabel("Count")
axes[2].set_title(f"Residual Distribution (MSE={final_mse:.4f})")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task11/task11_results.png", dpi=150, bbox_inches='tight')
print("Saved: task11/task11_results.png")

# --- Draw the PQC circuit ---
print("\nPQC Circuit Structure:")
drawer = qml.draw(quantum_circuit)
dummy_params = torch.zeros(N_PQC_PARAMS, dtype=torch.float64)
print(drawer(dummy_params))

fig2, ax2 = plt.subplots(1, 1, figsize=(16, 6))
circuit_str = drawer(dummy_params)
ax2.text(0.02, 0.5, circuit_str, transform=ax2.transAxes,
         fontsize=8, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax2.set_axis_off()
ax2.set_title("Parameterized Quantum Circuit (5 qubits, 2 layers)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("task11/task11_pqc_circuit.png", dpi=150, bbox_inches='tight')
print("Saved: task11/task11_pqc_circuit.png")
