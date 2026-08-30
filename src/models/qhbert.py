"""QHBERT model.

Split into two pieces on purpose:
  - QHBERTCore: bridge + quantum circuit + classifier head, operating on
    precomputed DistilBERT CLS embeddings (768-dim). This is the trainable
    part (~50K params) and is fast enough to train on CPU.
  - QHBERTFull: wraps a frozen DistilBERT encoder around QHBERTCore for
    end-to-end inference from raw text. Only needed where `transformers` +
    the pretrained weights are available (e.g. on Kaggle, or locally once
    you `pip install transformers`).

Training runs on cached embeddings using QHBERTCore directly — see
src/data/preprocess.py for the embedding-extraction step (intended to run
on Kaggle GPU, not locally).
"""

import torch
import torch.nn as nn

from .bridge import BridgeNetwork
from .quantum_circuit import QuantumLayer


class QHBERTCore(nn.Module):
    def __init__(self, bert_dim: int = 768, n_qubits: int = 8, n_layers: int = 3,
                 num_labels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bridge = BridgeNetwork(in_dim=bert_dim, n_qubits=n_qubits, dropout=dropout)
        self.quantum = QuantumLayer(n_qubits=n_qubits, n_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_labels),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        scaled = self.bridge(cls_embedding)
        quantum_out = self.quantum(scaled)
        return self.classifier(quantum_out)


class QHBERTFull(nn.Module):
    """End-to-end model: frozen DistilBERT -> QHBERTCore. Requires `transformers`."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3, num_labels: int = 2,
                 dropout: float = 0.3, bert_model: str = "distilbert-base-uncased"):
        super().__init__()
        from transformers import DistilBertModel

        self.bert = DistilBertModel.from_pretrained(bert_model)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.core = QHBERTCore(bert_dim=self.bert.config.dim, n_qubits=n_qubits,
                                n_layers=n_layers, num_labels=num_labels, dropout=dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_out.last_hidden_state[:, 0, :]
        return self.core(cls_embedding)


class QHBERTModel(nn.Module):
    """Configurable variant of QHBERTCore for the backend/qubit/layer ablation
    sweep: wider bridge (more trainable params than QHBERTCore's fixed ~50K)
    and a `backend` switch (PennyLane or Qiskit, via quantum_layers.build_quantum_layer)
    so qubits/layers/backend/feature-map/ansatz can vary per run without
    hand-editing a class. QHBERTCore stays the stable ~50K-param baseline;
    this is the wider comparison model built alongside it.
    """

    def __init__(
        self,
        bert_dim: int = 768,
        bridge_dims: tuple = (256, 64),
        n_qubits: int = 8,
        n_layers: int = 3,
        backend: str = "pennylane",
        head_dim: int = 64,
        num_labels: int = 2,
        dropout: float = 0.3,
        quantum_kwargs: dict | None = None,
    ):
        super().__init__()
        from src.models.quantum_layers import build_quantum_layer

        bridge_layers = []
        in_dim = bert_dim
        for dim in bridge_dims:
            bridge_layers += [nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = dim
        bridge_layers.append(nn.Linear(in_dim, n_qubits))
        self.bridge = nn.Sequential(*bridge_layers)

        self.quantum = build_quantum_layer(backend, n_qubits, n_layers, **(quantum_kwargs or {}))

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_labels),
        )

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        scaled = torch.tanh(self.bridge(cls_embedding)) * torch.pi
        # batched call, not a per-sample Python loop: both TorchLayer and TorchConnector
        # accept a full (batch, n_qubits) tensor directly, and looping multiplies each
        # backward pass's cost by batch_size for no benefit (measured ~74x slower on
        # PennyLane at 12 qubits/3 layers/batch 32 — see quantum_layers.py).
        quantum_out = self.quantum(scaled)
        return self.classifier(quantum_out)
