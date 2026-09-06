"""
gnn_model.py  [RESEARCH / NON-PRODUCTION]
──────────────────────────────────────────
GraphSAGE-style Graph Neural Network for exploratory research into
thermal propagation between rack micro-zones.

⚠️  THIS FILE IS NOT PART OF THE PRODUCTION RUNTIME PATH.
    Production uses the lightweight analytic GNN in src/features.py
    (the AnalyticGNN class), which has zero PyTorch dependency.

    This file is kept for future research phases (Phase 3+) where
    a learned spatial embedding may add value over the analytic proxy.

Dependencies (research env only):
    pip install torch torch-geometric
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn  import SAGEConv
import numpy as np
from typing import List, Tuple, Optional
import os


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_rack_graph(
    adjacency: List[Tuple[int, int]],
    node_features: torch.Tensor,          # shape: [n_racks, n_features]
    edge_weights: Optional[List[float]] = None,
) -> Data:
    """
    Convert a rack adjacency list into a PyG Data object.

    Parameters
    ──────────
    adjacency     : list of (src_rack, dst_rack) pairs (bidirectional).
    node_features : telemetry feature matrix, one row per rack.
    edge_weights  : optional proximity weights (0–1).
    """
    src, dst = zip(*adjacency)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    if edge_weights is not None:
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    return data


def linear_rack_adjacency(n_racks: int) -> List[Tuple[int, int]]:
    """Simple linear adjacency list for n racks in a row."""
    edges = []
    for i in range(n_racks - 1):
        edges.append((i,   i+1))
        edges.append((i+1, i  ))
    return edges


def grid_rack_adjacency(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Adjacency for a rows×cols rack grid (row-major IDs)."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    edges.append((node, nr * cols + nc))
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRAPHSAGE THERMAL MODEL  [Research-phase only]
# ─────────────────────────────────────────────────────────────────────────────

class ThermalGNN(nn.Module):
    """
    Two-layer GraphSAGE network that computes a thermal embedding per rack.

    Architecture:
      Layer 1  SAGEConv(in_feats → hidden_dim)  + ReLU + Dropout
      Layer 2  SAGEConv(hidden_dim → embed_dim)  + ReLU
      Risk head: Linear(embed_dim → 1)

    NOTE: Production uses AnalyticGNN in src/features.py.
    This is for validating whether learned embeddings add value over the
    analytic heat proxy (gnn_embedding = 0.7*self_heat + 0.3*neighbor_heat).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim:  int = 64,
        embed_dim:   int = 32,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.conv1    = SAGEConv(in_features, hidden_dim)
        self.conv2    = SAGEConv(hidden_dim,  embed_dim)
        self.dropout  = nn.Dropout(p=dropout)
        self.risk_head = nn.Linear(embed_dim, 1)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        embeddings  : [n_nodes, embed_dim]
        risk_scores : [n_nodes, 1]  (used for GNN pre-training loss)
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        embeddings  = F.relu(self.conv2(x, edge_index))
        risk_scores = self.risk_head(embeddings)
        return embeddings, risk_scores


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(
    model:        ThermalGNN,
    data:         Data,
    labels:       torch.Tensor,
    epochs:       int   = 200,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    verbose:      bool  = True,
) -> List[float]:
    """Pre-train the GNN on proxy thermal risk labels."""
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()
    history   = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad()
        _, risk_scores = model(data)
        loss = loss_fn(risk_scores.squeeze(), labels)
        loss.backward()
        optimiser.step()
        history.append(loss.item())
        if verbose and epoch % 20 == 0:
            print(f"  Epoch {epoch:>4d} | GNN MSE Loss: {loss.item():.4f}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE HELPER
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_thermal_embeddings(model: ThermalGNN, data: Data) -> np.ndarray:
    """Run trained GNN and return numpy embeddings for XGBoost."""
    model.eval()
    embeddings, _ = model(data)
    return embeddings.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  SAVE / LOAD
# ─────────────────────────────────────────────────────────────────────────────

def save_gnn(model: ThermalGNN, path: str):
    torch.save(model.state_dict(), path)
    print(f"GNN saved → {path}")


def load_gnn(path: str, in_features: int, hidden_dim=64, embed_dim=32) -> ThermalGNN:
    model = ThermalGNN(in_features, hidden_dim, embed_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"GNN loaded ← {path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[RESEARCH] 4-node thermal GNN smoke test …")
    node_feats = torch.tensor([
        [30.0, 25.0, 50.0, 20.0, 15.0],
        [95.0, 90.0, 88.0, 75.0, 60.0],
        [35.0, 30.0, 55.0, 22.0, 18.0],
        [20.0, 15.0, 40.0, 10.0, 12.0],
    ], dtype=torch.float)

    adj   = linear_rack_adjacency(n_racks=4)
    graph = build_rack_graph(adj, node_feats)
    labels = torch.tensor([35.0, 88.0, 40.0, 20.0])

    model = ThermalGNN(in_features=5, hidden_dim=16, embed_dim=8)
    train_gnn(model, graph, labels, epochs=50, verbose=False)

    embs = get_thermal_embeddings(model, graph)
    print(f"Embedding shape: {embs.shape}  (expected [4, 8])")
    print("training/research/gnn_model.py  ✓ [RESEARCH ONLY]")
