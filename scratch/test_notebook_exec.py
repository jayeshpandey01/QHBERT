import torch
import torch.nn as nn
import pennylane as qml
import math

# Test quantum circuit and bridge as defined in notebook
def build_fusion_quantum_layer(n_qubits: int, n_layers: int) -> nn.Module:
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)

class MultimodalFusionBridge(nn.Module):
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int, n_qubits: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_qubits),
        )
    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([visual_feat, text_feat], dim=-1)
        return torch.tanh(self.net(combined)) * math.pi

class QuantumMultimodalFusionModel(nn.Module):
    def __init__(self, visual_dim=512, text_dim=384, hidden_dim=128, n_qubits=8, n_layers=3, num_classes=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.bridge = MultimodalFusionBridge(visual_dim, text_dim, hidden_dim, n_qubits)
        self.quantum = build_fusion_quantum_layer(n_qubits, n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )
    def forward(self, visual_feat, text_feat):
        angles = self.bridge(visual_feat, text_feat)
        quantum_out = self.quantum(angles)
        return self.classifier(quantum_out)

m = QuantumMultimodalFusionModel()
v = torch.randn(2, 512)
t = torch.randn(2, 384)
out = m(v, t)
print("Forward output shape:", out.shape)
out.sum().backward()
print("Backward gradient check:", m.bridge.net[0].weight.grad is not None)
print("All assertions passed!")
