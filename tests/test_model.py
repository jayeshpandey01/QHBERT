import torch

from src.models import BridgeNetwork, QHBERTCore, QuantumLayer


def test_bridge_output_range():
    bridge = BridgeNetwork(in_dim=768, n_qubits=8)
    x = torch.randn(4, 768)
    out = bridge(x)
    assert out.shape == (4, 8)
    assert out.min() >= -torch.pi and out.max() <= torch.pi


def test_quantum_layer_output_shape():
    layer = QuantumLayer(n_qubits=8, n_layers=3)
    x = torch.rand(8) * torch.pi
    out = layer(x)
    assert out.shape == (8,)
    assert torch.all(out >= -1.0001) and torch.all(out <= 1.0001)


def test_qhbert_core_forward_and_backward():
    model = QHBERTCore(bert_dim=768, n_qubits=8, n_layers=3, num_labels=2)
    x = torch.randn(3, 768)
    logits = model(x)
    assert logits.shape == (3, 2)

    loss = logits.sum()
    loss.backward()
    assert model.bridge.net[0].weight.grad is not None


def test_qhbert_core_param_count():
    model = QHBERTCore()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 50090
