"""Classical baseline models for the fake-news comparison table.

Architectures adapted from src/notebook/a4-fakenews-complete-ml-ann-transformer-analysis.ipynb
(a third-party Kaggle notebook), re-implemented in PyTorch instead of
Keras/TensorFlow to match this repo's stack (QHBERTCore, quantum circuits)
and keep the local dev environment on one framework. Hyperparameters
(embedding_dim=32, hidden sizes, etc.) are the values that notebook's own
sweep experiments found best — not re-tuned here.

Two real fixes vs. the source notebook, not carried over:
  1. It mixes an unrelated COVID-19 tweets dataset into training partway
     through (a tweet-vs-article confound). These builders are dataset-
     agnostic; the caller is responsible for a single clean split.
  2. ISOT/WELFake are known "easy" benchmarks (source/style leakage lets
     any model hit ~99%) — that's a property of the data, not evidence
     these architectures are unusually strong. Report accordingly.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> Dense -> Dropout -> Dense(num_classes)."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.3, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # hidden: (num_directions, batch, hidden_dim) -> concat forward+backward
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.classifier(last_hidden)


class CNNClassifier(nn.Module):
    """Embedding -> Conv1d -> GlobalMaxPool -> Dense -> Dropout -> Dense(num_classes)."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, num_filters: int = 128,
                 kernel_size: int = 5, hidden_dim: int = 64, num_classes: int = 2,
                 dropout: float = 0.3, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(num_filters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).transpose(1, 2)  # (batch, embedding_dim, seq_len)
        conv_out = self.conv(embedded)
        return self.classifier(conv_out)


class TransformerEncoderClassifier(nn.Module):
    """Embedding + positional -> TransformerEncoderLayer -> GlobalAvgPool -> Dense -> Dense(num_classes)."""

    def __init__(self, vocab_size: int, max_len: int = 200, embedding_dim: int = 64,
                 num_heads: int = 4, ff_dim: int = 128, hidden_dim: int = 64,
                 num_classes: int = 2, dropout: float = 0.2, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        embedded = self.embedding(x) + self.pos_embedding(positions)
        encoded = self.encoder(embedded)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def build_tfidf_svm():
    """TF-IDF + LinearSVC pipeline, matching the source notebook's Model 1."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])
