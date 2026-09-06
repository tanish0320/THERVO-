"""
gnn_model.py
────────────
GraphSAGE-style Graph Neural Network that models thermal propagation
between rack micro-zones in a sensor-free data-centre cooling system.

Each node  = one rack (micro-zone)
Each edge  = physical adjacency / heat-influence path between racks

The GNN produces a thermal embedding per rack that encodes BOTH the
rack's own workload state AND the thermal pressure arriving from its
neighbours — something a single-rack model cannot capture.

Dependencies
────────────
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
    adjacency     : list of (src_rack, dst_rack) pairs.
                    Heat flows BOTH ways → add both directions.
    node_features : telemetry feature matrix, one row per rack.
    edge_weights  : optional physical proximity / shared-wall weights (0–1).
                    Closer racks share more heat → higher weight.

    Returns
    ───────
    PyG Data object ready for GNN forward pass.

    Example – 4-rack linear layout (rack 0 next to 1, 1 next to 2, …)
    ──────────────────────────────────────────────────────────────────
    adjacency = [(0,1),(1,0),(1,2),(2,1),(2,3),(3,2)]
    """
    # Build bidirectional edge_index  [2, num_edges]
    src, dst = zip(*adjacency)
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)

    if edge_weights is not None:
        data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    return data


def linear_rack_adjacency(n_racks: int) -> List[Tuple[int, int]]:
    """
    Helper: build a simple linear adjacency list for n racks in a row.
    Each rack neighbours the one immediately to its left and right.
    """
    edges = []
    for i in range(n_racks - 1):
        edges.append((i,   i+1))
        edges.append((i+1, i  ))
    return edges


def grid_rack_adjacency(rows: int, cols: int) -> List[Tuple[int, int]]:
    """
    Helper: build adjacency for a rows×cols rack grid.
    Rack IDs are row-major: rack(r,c) = r*cols + c.
    """
    edges = []
    for r in range(rows):
        for c in range(cols):
            node = r * cols + c
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbour = nr * cols + nc
                    edges.append((node, neighbour))
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRAPHSAGE THERMAL MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ThermalGNN(nn.Module):
    """
    Two-layer GraphSAGE network that computes a thermal embedding for each rack.

    Architecture
    ────────────
    Layer 1  SAGEConv(in_feats  → hidden_dim)  + ReLU + Dropout
    Layer 2  SAGEConv(hidden_dim → embed_dim)   + ReLU

    GraphSAGE aggregation at each layer:
        h_v^(l) = σ( W · CONCAT( h_v^(l-1),  MEAN_{u∈N(v)} h_u^(l-1) ) )

    This implements:
        embedding_i = f( self_features + neighbour_features )

    The mean aggregation over neighbours models how heat diffuses from
    adjacent racks — a rack surrounded by heavily loaded neighbours will
    receive a higher thermal embedding even if its own load is moderate.

    Parameters
    ──────────
    in_features  : number of input features per node (engineered telemetry)
    hidden_dim   : internal representation width (default 64)
    embed_dim    : output embedding dimension passed to XGBoost (default 32)
    dropout      : regularisation (default 0.3)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim:  int = 64,
        embed_dim:   int = 32,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.conv1   = SAGEConv(in_features, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim,  embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # Thermal risk head: scalar per node (used during GNN training)
        self.risk_head = nn.Linear(embed_dim, 1)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        embeddings  : [n_nodes, embed_dim]  — fed into XGBoost
        risk_scores : [n_nodes, 1]          — used for GNN pre-training loss
        """
        x, edge_index = data.x, data.edge_index

        # Layer 1: aggregate self + neighbour features
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: refine thermal embedding
        x = self.conv2(x, edge_index)
        embeddings = F.relu(x)                        # [n_nodes, embed_dim]

        risk_scores = self.risk_head(embeddings)      # [n_nodes, 1]
        return embeddings, risk_scores


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP (pre-train GNN on thermal risk proxy)
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(
    model:       ThermalGNN,
    data:        Data,
    labels:      torch.Tensor,          # [n_nodes] thermal risk scores
    epochs:      int   = 200,
    lr:          float = 1e-3,
    weight_decay: float = 1e-4,
    verbose:     bool  = True,
) -> List[float]:
    """
    Train the GNN using MSE loss against thermal risk labels.

    In real deployment labels come from:
      • Stress-test ground truth (Prime95 / FurMark runs)
      • OR the composite proxy from data_processing.generate_thermal_risk_label()

    Returns loss history for plotting / early-stopping analysis.
    """
    optimiser = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
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
def get_thermal_embeddings(
    model: ThermalGNN,
    data:  Data,
) -> np.ndarray:
    """
    Run a trained GNN and return numpy embeddings ready for XGBoost.

    Returns
    ───────
    embeddings : np.ndarray of shape [n_racks, embed_dim]
    """
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
# 6.  WORKED EXAMPLE  (4-node graph)
# ─────────────────────────────────────────────────────────────────────────────

def demo_4node_graph():
    """
    Demonstrates heat propagation on a 4-rack linear cluster.

    Layout:  [Rack-0] ── [Rack-1] ── [Rack-2] ── [Rack-3]

    Rack-1 is under heavy load (cpu=95, gpu=90).
    We expect Rack-0 and Rack-2 to pick up elevated thermal embeddings
    from GNN neighbour aggregation even if their own load is moderate.
    """
    print("\n" + "="*60)
    print("  4-NODE THERMAL GNN DEMO")
    print("="*60)

    # ── Node features (5 raw telemetry values: cpu,gpu,mem,disk,net) ──
    #    Rack-1 deliberately stressed
    node_feats = torch.tensor([
        [30.0, 25.0, 50.0, 20.0, 15.0],   # Rack 0 – moderate
        [95.0, 90.0, 88.0, 75.0, 60.0],   # Rack 1 – HOT  ←
        [35.0, 30.0, 55.0, 22.0, 18.0],   # Rack 2 – moderate
        [20.0, 15.0, 40.0, 10.0, 12.0],   # Rack 3 – cool
    ], dtype=torch.float)

    # ── Linear adjacency: 0↔1, 1↔2, 2↔3 ──
    adj = linear_rack_adjacency(n_racks=4)
    graph = build_rack_graph(adj, node_feats)

    # ── Synthetic thermal risk labels (0–100) ──
    labels = torch.tensor([35.0, 88.0, 40.0, 20.0])

    # ── Instantiate and train GNN ──
    in_feats = node_feats.shape[1]   # 5
    model = ThermalGNN(in_features=in_feats, hidden_dim=16, embed_dim=8)

    print("\nPre-training GNN (200 epochs) …")
    train_gnn(model, graph, labels, epochs=200, lr=5e-3, verbose=True)

    # ── Extract embeddings ──
    embeddings = get_thermal_embeddings(model, graph)
    print("\nThermal Embeddings per rack (shape:", embeddings.shape, ")")
    for i, emb in enumerate(embeddings):
        print(f"  Rack {i}: [{', '.join(f'{v:+.3f}' for v in emb)}]")

    # ── Verify neighbour influence ──
    print("\n[HEAT PROPAGATION CHECK]")
    print("Rack-1 is stressed. Rack-0 and Rack-2 share an edge with it.")
    norms = np.linalg.norm(embeddings, axis=1)
    print("Embedding L2 norms (higher = more thermal pressure):")
    for i, n in enumerate(norms):
        bar = "█" * int(n * 4)
        print(f"  Rack {i}: {n:.3f}  {bar}")

    return model, graph, embeddings


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, graph, embeddings = demo_4node_graph()
    print("\ngnn_model.py  ✓")
