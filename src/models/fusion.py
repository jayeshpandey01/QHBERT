"""Quantum Multimodal & Hybrid Fusion Model.

Bridges pre-trained vision extractors (ResNet18 / QPIE) and pre-trained
text transformers (DistilBERT / MiniLM) through a classical dimensionality-reduction
fusion bridge into a shared Variational Quantum Circuit (VQC).

Supports:
  1. Multimodal Early/Feature-Level Fusion: [visual_emb || text_emb] -> Bridge -> VQC -> Classifier Head
  2. Modality Ablations: Text-Only, Image-Only, and Multimodal Fusion
  3. Classical Baseline: Drop-in classical MLP replacement for the quantum layer to measure quantum advantage
  4. Quantum State & Angle Explainability: inspectable rotation angles and Pauli-Z expectations
"""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .quantum_layers import build_pennylane_layer


class MultimodalFusionBridge(nn.Module):
    """Fuses visual and textual representation vectors, compresses to n_qubits dimensions,
    and applies tanh(x) * pi scaling to map directly to the single-qubit rotation domain [-pi, pi].
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 384,
        hidden_dim: int = 128,
        n_qubits: int = 8,
        dropout: float = 0.3,
        fusion_type: Literal["concat", "gated"] = "concat",
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.fusion_type = fusion_type
        self.n_qubits = n_qubits

        if fusion_type == "gated":
            # Learnable gating mechanism: g = sigmoid(W_g * [v || t])
            # fused = g * W_v(v) + (1 - g) * W_t(t)
            self.visual_proj = nn.Linear(visual_dim, hidden_dim)
            self.text_proj = nn.Linear(text_dim, hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(visual_dim + text_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            in_features = hidden_dim
        else:
            # Standard concatenation fusion: [v || t]
            in_features = visual_dim + text_dim

        self.projection = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "gated":
            v_p = self.visual_proj(visual_feat)
            t_p = self.text_proj(text_feat)
            g = self.gate(torch.cat([visual_feat, text_feat], dim=-1))
            fused = g * v_p + (1.0 - g) * t_p
            reduced = self.projection(fused)
        else:
            combined = torch.cat([visual_feat, text_feat], dim=-1)
            reduced = self.projection(combined)

        return torch.tanh(reduced) * math.pi


class QuantumMultimodalFusionCore(nn.Module):
    """Core trainable stack: Multimodal Bridge -> Parameterized Quantum Circuit -> Classifier Head.

    Operates on precomputed feature tensors, making it extremely fast to train on CPU
    or GPU without repetitive forward passes through heavy vision/language backbones.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 384,
        hidden_dim: int = 128,
        n_qubits: int = 8,
        n_layers: int = 3,
        num_classes: int = 2,
        head_dim: int = 32,
        dropout: float = 0.3,
        fusion_type: Literal["concat", "gated"] = "concat",
        backend: str = "pennylane",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        self.bridge = MultimodalFusionBridge(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            dropout=dropout,
            fusion_type=fusion_type,
        )

        if backend == "pennylane":
            self.quantum = build_pennylane_layer(n_qubits, n_layers)
        elif backend == "qiskit":
            from .quantum_layers import build_qiskit_layer
            self.quantum = build_qiskit_layer(n_qubits, n_layers)
        else:
            raise ValueError(f"Unknown quantum backend: {backend}")

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        angles = self.bridge(visual_feat, text_feat)
        # Device-safe round-trip: PennyLane default.qubit simulator executes on CPU,
        # while bridge and classifier run on CUDA GPU (DEVICE).
        # .cpu() and .to() are autograd-differentiable, preserving exact backward gradients.
        quantum_out = self.quantum(angles.cpu()).to(angles.device)
        return self.classifier(quantum_out)

    def get_quantum_angles(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Returns the [batch, n_qubits] rotation angles in [-pi, pi] for explainability inspection."""
        with torch.no_grad():
            return self.bridge(visual_feat, text_feat)

    def get_circuit_expectations(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Returns the [batch, n_qubits] Pauli-Z expectation values in [-1, 1]."""
        with torch.no_grad():
            angles = self.bridge(visual_feat, text_feat)
            return self.quantum(angles.cpu()).to(angles.device)


class ClassicalMultimodalFusionBaseline(nn.Module):
    """Classical control baseline: replaces the quantum circuit with an identity/linear layer
    having identical or equivalent parameter budget to measure true quantum advantage.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        text_dim: int = 384,
        hidden_dim: int = 128,
        latent_dim: int = 8,
        head_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_type: Literal["concat", "gated"] = "concat",
    ):
        super().__init__()
        self.bridge = MultimodalFusionBridge(
            visual_dim=visual_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            n_qubits=latent_dim,
            dropout=dropout,
            fusion_type=fusion_type,
        )
        self.classical_core = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        latent = self.classical_core(self.bridge(visual_feat, text_feat))
        return self.classifier(latent)


class QuantumMultimodalFusionFull(nn.Module):
    """End-to-end model wrapping pre-trained ResNet18 and pre-trained text encoder
    with QuantumMultimodalFusionCore.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 3,
        num_classes: int = 2,
        text_dim: int = 384,
        dropout: float = 0.3,
        freeze_vision: bool = True,
        backend: str = "pennylane",
    ):
        super().__init__()
        import torchvision.models as tv_models

        # Pretrained ResNet18
        self.backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Outputs 512-dim embedding

        if freeze_vision:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.core = QuantumMultimodalFusionCore(
            visual_dim=512,
            text_dim=text_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            num_classes=num_classes,
            dropout=dropout,
            backend=backend,
        )

    def unfreeze_backbone_layer4(self):
        """Unfreezes ResNet18 layer4 for Phase 2 joint fine-tuning."""
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        visual_feats = self.backbone(images)
        return self.core(visual_feats, text_embeds)
