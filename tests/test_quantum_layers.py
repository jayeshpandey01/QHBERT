import torch

from src.models.quantum_layers import build_pennylane_layer, build_qiskit_layer
from src.models.qhbert import QHBERTModel

N_QUBITS = 4
N_LAYERS = 2
BATCH = 3


def test_pennylane_layer_forward_and_backward():
    layer = build_pennylane_layer(N_QUBITS, N_LAYERS)
    x = torch.rand(BATCH, N_QUBITS, requires_grad=True)
    out = torch.stack([layer(x[i]) for i in range(BATCH)])
    assert out.shape == (BATCH, N_QUBITS)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_qiskit_layer_forward_and_backward():
    layer = build_qiskit_layer(N_QUBITS, N_LAYERS)
    x = torch.rand(BATCH, N_QUBITS, requires_grad=True)
    out = layer(x)
    assert out.shape == (BATCH, N_QUBITS)
    out.sum().backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)


def test_qiskit_layer_variants_run():
    for feature_map in ("zz", "pauli"):
        for ansatz in ("real_amplitudes", "efficient_su2"):
            layer = build_qiskit_layer(N_QUBITS, N_LAYERS, feature_map=feature_map, ansatz=ansatz)
            out = layer(torch.rand(2, N_QUBITS))
            assert out.shape == (2, N_QUBITS)


def test_qhbert_model_both_backends():
    for backend in ("pennylane", "qiskit"):
        model = QHBERTModel(
            bert_dim=32, bridge_dims=(16,), n_qubits=N_QUBITS, n_layers=N_LAYERS, backend=backend
        )
        x = torch.rand(BATCH, 32)
        logits = model(x)
        assert logits.shape == (BATCH, 2)
        logits.sum().backward()
        bridge_grad = next(model.bridge.parameters()).grad
        assert bridge_grad is not None and torch.any(bridge_grad != 0)
