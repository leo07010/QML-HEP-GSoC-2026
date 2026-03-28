"""
Task XII: RL-based MLP + PQC Embedding

Same embedding task as Task XI, but trained with PPO instead of supervised backprop.
- State:  normally distributed input vector (dim=8)
- Action: PQC rotation parameters (dim=30), continuous
- Reward: -MSE between PQC output <Z> and target expectations
- Algorithm: Proximal Policy Optimization (PPO)

Framework: PyTorch + PennyLane
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import pennylane as qml
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# 1. PQC (same as Task XI)
# ============================================================
N_QUBITS = 5
N_LAYERS = 2
N_PQC_PARAMS = N_QUBITS * N_LAYERS * 3  # 30

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(params):
    params = params.reshape(N_LAYERS, N_QUBITS, 3)
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RX(params[layer, q, 0], wires=q)
            qml.RY(params[layer, q, 1], wires=q)
            qml.RZ(params[layer, q, 2], wires=q)
        for q in range(N_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]


# ============================================================
# 2. Data Generation (same as Task XI)
# ============================================================
INPUT_DIM = 8

np.random.seed(0)
W_target = np.random.randn(INPUT_DIM, N_QUBITS).astype(np.float64) * 0.5
b_target = np.random.randn(N_QUBITS).astype(np.float64) * 0.1
np.random.seed(42)


def generate_data(n_samples):
    X = np.random.randn(n_samples, INPUT_DIM).astype(np.float64)
    y = np.tanh(X @ W_target + b_target)
    return torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)


N_TRAIN = 500
N_TEST = 100
X_train, y_train = generate_data(N_TRAIN)
X_test, y_test = generate_data(N_TEST)

print(f"Qubits: {N_QUBITS}, PQC params: {N_PQC_PARAMS}")
print(f"Train: {N_TRAIN}, Test: {N_TEST}")


# ============================================================
# 3. PPO Agent (Actor-Critic)
# ============================================================
class Actor(nn.Module):
    """Policy network: state -> mean & log_std of PQC parameters."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float64),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(64, action_dim, dtype=torch.float64)
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float64))

    def forward(self, state):
        features = self.net(state)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, state, action):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Value network: state -> V(s)."""
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 64, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(64, 1, dtype=torch.float64),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


def compute_reward(actions, targets):
    """
    Per-qubit reward shaping: reward = sum of -|pred_q - target_q|^2.
    Clearer credit assignment than single scalar MSE.
    """
    rewards = []
    for i in range(actions.shape[0]):
        exp_vals = quantum_circuit(actions[i])
        pqc_out = torch.stack(exp_vals)
        per_qubit_errors = (pqc_out - targets[i]) ** 2
        reward = -per_qubit_errors.sum()
        rewards.append(reward.detach())
    return torch.stack(rewards)


# ============================================================
# 4. PPO Training
# ============================================================
actor = Actor(INPUT_DIM, N_PQC_PARAMS)
critic = Critic(INPUT_DIM)
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-3)

# PPO hyperparameters
EPOCHS = 2000
BATCH_SIZE = 64
PPO_CLIP = 0.2
PPO_UPDATES = 8
ENTROPY_COEF = 0.005

actor_params = sum(p.numel() for p in actor.parameters())
critic_params = sum(p.numel() for p in critic.parameters())
print(f"\nActor params:  {actor_params:,}")
print(f"Critic params: {critic_params:,}")
print(f"PPO clip: {PPO_CLIP}, updates/epoch: {PPO_UPDATES}")
print(f"\nTraining for {EPOCHS} epochs, batch_size={BATCH_SIZE}")
print("-" * 60)

train_rewards = []
test_mses = []


def evaluate_test(actor):
    """Evaluate on test set: compute mean MSE."""
    actor.eval()
    with torch.no_grad():
        mean, _ = actor(X_test)
    mses = []
    for i in range(N_TEST):
        exp_vals = quantum_circuit(mean[i])
        pqc_out = torch.stack(exp_vals)
        mse = F.mse_loss(pqc_out, y_test[i])
        mses.append(mse.item())
    actor.train()
    return np.mean(mses)


for epoch in range(1, EPOCHS + 1):
    # Sample a batch of states
    idx = np.random.choice(N_TRAIN, size=BATCH_SIZE, replace=False)
    states = X_train[idx]
    targets = y_train[idx]

    # Collect trajectories
    with torch.no_grad():
        actions, old_log_probs = actor.get_action(states)
        rewards = compute_reward(actions, targets)
        values = critic(states)
        advantages = rewards - values

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO update
    for _ in range(PPO_UPDATES):
        # Actor update
        new_log_probs, entropy = actor.evaluate(states, actions)
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()

        actor_optim.zero_grad()
        actor_loss.backward()
        actor_optim.step()

        # Critic update
        v_pred = critic(states)
        critic_loss = F.mse_loss(v_pred, rewards)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()

    avg_reward = rewards.mean().item()
    train_rewards.append(avg_reward)

    # Evaluate on test set periodically
    if epoch % 5 == 0 or epoch == 1:
        test_mse = evaluate_test(actor)
        test_mses.append((epoch, test_mse))
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Test MSE: {test_mse:.6f}")

# Final evaluation
final_test_mse = evaluate_test(actor)
print(f"\nFinal Test MSE: {final_test_mse:.6f}")

# ============================================================
# 5. Comparison with Task XI (supervised)
# ============================================================
task11_mse = 0.012039  # from Task XI results

print(f"\n{'='*50}")
print("COMPARISON: Supervised vs RL")
print(f"{'='*50}")
print(f"{'Method':<25} {'Test MSE':>10}")
print(f"{'-'*35}")
print(f"{'Task XI (Supervised)':<25} {task11_mse:>10.6f}")
print(f"{'Task XII (PPO)':<25} {final_test_mse:>10.6f}")

# ============================================================
# 6. Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training reward curve
axes[0].plot(train_rewards, alpha=0.3, color='blue')
# Smoothed
window = 10
if len(train_rewards) > window:
    smoothed = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, len(train_rewards)), smoothed, color='blue', linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Average Reward (-MSE)")
axes[0].set_title("PPO Training Reward")
axes[0].grid(True, alpha=0.3)

# Test MSE over time
test_epochs, test_vals = zip(*test_mses)
axes[1].plot(test_epochs, test_vals, 'o-', color='orange')
axes[1].axhline(y=task11_mse, color='green', linestyle='--', label=f'Task XI (Supervised): {task11_mse:.4f}')
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Test MSE")
axes[1].set_title("Test MSE: PPO vs Supervised Baseline")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Predicted vs Target (final)
actor.eval()
with torch.no_grad():
    final_mean, _ = actor(X_test)
final_pred = []
for i in range(N_TEST):
    exp_vals = quantum_circuit(final_mean[i])
    final_pred.append(torch.stack(exp_vals).detach().numpy())
final_pred = np.array(final_pred)
final_target = y_test.numpy()

for q in range(N_QUBITS):
    axes[2].scatter(final_target[:, q], final_pred[:, q], alpha=0.4, s=10, label=f"Q{q}")
axes[2].plot([-1, 1], [-1, 1], "k--", alpha=0.5)
axes[2].set_xlabel("Target <Z>")
axes[2].set_ylabel("Predicted <Z>")
axes[2].set_title("PPO: Predicted vs Target (per qubit)")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(-1.1, 1.1)
axes[2].set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("task12/task12_results.png", dpi=150, bbox_inches='tight')
print("\nSaved: task12/task12_results.png")
