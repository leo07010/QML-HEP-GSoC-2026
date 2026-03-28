"""
Task II: Classical Graph Neural Network (GNN)
Quark/Gluon jet classification using two graph-based architectures:
  1. EdgeConv (Dynamic Graph CNN) — inspired by ParticleNet
  2. GATv2 (Graph Attention Network v2)

Dataset: ParticleNet Quark/Gluon from Zenodo (QG_jets.npz)
  X: (N, 139, 4) — up to 139 particles per jet, 4 features per particle
     Features: (eta, phi, pt, pdgid)
  y: (N,) — 0=gluon, 1=quark
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import EdgeConv, GATv2Conv, global_mean_pool, knn_graph
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Data Loading & Graph Construction
# ============================================================
print("Loading data...")
raw = np.load("task2/data/QG_jets.npz")
X_all, y_all = raw["X"], raw["y"]

# Use a subset for tractability (20k train, 5k val, 5k test)
N_TRAIN, N_VAL, N_TEST = 20000, 5000, 5000
X_train, y_train = X_all[:N_TRAIN], y_all[:N_TRAIN]
X_val, y_val = X_all[N_TRAIN:N_TRAIN+N_VAL], y_all[N_TRAIN:N_TRAIN+N_VAL]
X_test, y_test = X_all[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST], y_all[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST]

print(f"Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")
print(f"Label distribution (train): quark={y_train.sum():.0f}, gluon={N_TRAIN - y_train.sum():.0f}")

K_NEIGHBORS = 7  # number of nearest neighbors for graph construction


def build_graph_data(X, y, k=K_NEIGHBORS):
    """
    Convert point-cloud jet data to PyG graph list.

    Graph construction strategy:
    - Each particle in a jet becomes a node.
    - Zero-padded particles (all features = 0) are removed.
    - Node features: (eta, phi, pt_log, pdgid_encoded)
      * pt is log-transformed for better scale handling
      * pdgid is kept as-is (categorical feature)
    - Edges are built using k-nearest neighbors in (eta, phi) space,
      which respects the angular geometry of particle detectors.
      Nearby particles in eta-phi space are physically related
      (e.g., from the same shower/subjet).
    """
    data_list = []
    for i in range(len(X)):
        jet = X[i]  # (139, 4): eta, phi, pt, pdgid

        # Remove zero-padded particles
        mask = np.any(jet != 0, axis=1)
        jet = jet[mask]

        if len(jet) < 2:
            continue

        eta = jet[:, 0]
        phi = jet[:, 1]
        pt = jet[:, 2]
        pdgid = jet[:, 3]

        # Feature engineering
        pt_log = np.log1p(np.abs(pt))  # log(1 + |pt|) for scale normalization

        # Node features: eta, phi, pt_log, pdgid
        node_features = np.stack([eta, phi, pt_log, pdgid], axis=1).astype(np.float32)

        x = torch.tensor(node_features, dtype=torch.float)
        label = torch.tensor([y[i]], dtype=torch.long)

        # Build kNN graph in (eta, phi) space
        coords = torch.tensor(np.stack([eta, phi], axis=1), dtype=torch.float)
        edge_index = knn_graph(coords, k=min(k, len(jet) - 1), loop=False)

        data_list.append(Data(x=x, edge_index=edge_index, y=label))

    return data_list


print("Building graphs...")
t0 = time.time()
train_data = build_graph_data(X_train, y_train)
val_data = build_graph_data(X_val, y_val)
test_data = build_graph_data(X_test, y_test)
print(f"Graph construction: {time.time()-t0:.1f}s")
print(f"Example graph: {train_data[0]}")

BATCH_SIZE = 256
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 2. Model 1: EdgeConv (ParticleNet-style Dynamic Graph CNN)
# ============================================================
class EdgeConvNet(nn.Module):
    """
    Dynamic EdgeConv model inspired by ParticleNet.
    Uses edge convolutions that learn edge features from node pairs,
    allowing the model to capture local geometric structure.
    """
    def __init__(self, in_channels=4, hidden=64):
        super().__init__()
        # EdgeConv blocks with increasing capacity
        self.conv1 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            ), aggr='mean'
        )
        self.conv2 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            ), aggr='mean'
        )
        self.conv3 = EdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden, hidden * 2),
                nn.BatchNorm1d(hidden * 2),
                nn.ReLU(),
                nn.Linear(hidden * 2, hidden * 2),
            ), aggr='mean'
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)

        # Global pooling
        x = global_mean_pool(x, batch)

        return self.classifier(x)


# ============================================================
# 3. Model 2: GATv2 (Graph Attention Network v2)
# ============================================================
class GATv2Net(nn.Module):
    """
    GATv2 model for jet classification.
    Uses attention mechanisms to learn which neighboring particles
    are most important for classification, providing interpretability.
    Multi-head attention captures different types of particle relationships.
    """
    def __init__(self, in_channels=4, hidden=64, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden, heads=heads, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden * heads)
        self.conv2 = GATv2Conv(hidden * heads, hidden, heads=heads, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden * heads)
        self.conv3 = GATv2Conv(hidden * heads, hidden * 2, heads=1, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.relu(self.bn3(self.conv3(x, edge_index)))

        x = global_mean_pool(x, batch)

        return self.classifier(x)


# ============================================================
# 4. Training & Evaluation
# ============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    for data in loader:
        data = data.to(DEVICE)
        out = model(data)
        probs = F.softmax(out, dim=1)[:, 1]
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.y.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(data.y.squeeze().cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    acc = correct / total
    auc = roc_auc_score(all_labels, all_probs)
    return acc, auc, all_probs, all_labels


def run_experiment(model_class, model_name, epochs=30, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    model = model_class().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc, val_auc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"task2/{model_name}_best.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"Val AUC: {val_auc:.4f} | "
                  f"{time.time()-t0:.1f}s")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(f"task2/{model_name}_best.pt", weights_only=True))
    test_acc, test_auc, test_probs, test_labels = evaluate(model, test_loader)
    print(f"\n  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    return history, test_acc, test_auc, test_probs, test_labels


# Run both experiments
h1, acc1, auc1, probs1, labels1 = run_experiment(EdgeConvNet, "EdgeConv", epochs=30)
h2, acc2, auc2, probs2, labels2 = run_experiment(GATv2Net, "GATv2", epochs=30)

# ============================================================
# 5. Comparison & Visualization
# ============================================================
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
print(f"{'Model':<15} {'Test Acc':>10} {'Test AUC':>10}")
print(f"{'-'*35}")
print(f"{'EdgeConv':<15} {acc1:>10.4f} {auc1:>10.4f}")
print(f"{'GATv2':<15} {acc2:>10.4f} {auc2:>10.4f}")

# Plot training curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss
axes[0].plot(h1["train_loss"], label="EdgeConv")
axes[0].plot(h2["train_loss"], label="GATv2")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Training Loss")
axes[0].set_title("Training Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Validation AUC
axes[1].plot(h1["val_auc"], label="EdgeConv")
axes[1].plot(h2["val_auc"], label="GATv2")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].set_title("Validation AUC")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ROC Curves
fpr1, tpr1, _ = roc_curve(labels1, probs1)
fpr2, tpr2, _ = roc_curve(labels2, probs2)
axes[2].plot(fpr1, tpr1, label=f"EdgeConv (AUC={auc1:.4f})")
axes[2].plot(fpr2, tpr2, label=f"GATv2 (AUC={auc2:.4f})")
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve (Test Set)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("task2/task2_results.png", dpi=150, bbox_inches='tight')
print("\nSaved: task2/task2_results.png")
