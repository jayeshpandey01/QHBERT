"""Classical dimensionality-reduction bridge: BERT embedding -> quantum-scale features."""

import torch
import torch.nn as nn


class BridgeNetwork(nn.Module):
    """Reduces a 768-dim DistilBERT CLS embedding to n_qubits dims, scaled to [-pi, pi]."""

    def __init__(self, in_dim: int = 768, hidden_dim: int = 64, n_qubits: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.net(x)
        return torch.tanh(reduced) * torch.pi
