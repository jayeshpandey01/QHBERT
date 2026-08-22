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
        quantum_out = torch.stack([self.quantum(scaled[i]) for i in range(scaled.shape[0])])
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
