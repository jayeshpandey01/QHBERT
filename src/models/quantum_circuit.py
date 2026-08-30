"""QHBERT quantum circuit: 8-qubit strongly entangling variational circuit."""

import torch
import torch.nn as nn

from src.models.quantum_layers import build_pennylane_layer


class QuantumLayer(nn.Module):
    """Parameterized quantum circuit as a PyTorch layer.

    AngleEmbedding -> n_layers x (ring CNOT entangler + RY/RZ rotations) -> Pauli-Z readout.

    Circuit definition lives in quantum_layers.build_pennylane_layer (shared
    with the qubit/layer ablation sweep) so there's one place to change it.
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qlayer = build_pennylane_layer(n_qubits, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qlayer(x)
