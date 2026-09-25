import torch
import pytest

from src.models import (
    MultimodalFusionBridge,
    QuantumMultimodalFusionCore,
    ClassicalMultimodalFusionBaseline,
)


def test_fusion_bridge_output_range():
    bridge = MultimodalFusionBridge(visual_dim=512, text_dim=384, n_qubits=8)
    v = torch.randn(4, 512)
    t = torch.randn(4, 384)
    out = bridge(v, t)
    assert out.shape == (4, 8)
    assert out.min() >= -torch.pi and out.max() <= torch.pi


def test_fusion_bridge_gated():
    bridge = MultimodalFusionBridge(visual_dim=512, text_dim=384, n_qubits=8, fusion_type="gated")
    v = torch.randn(4, 512)
    t = torch.randn(4, 384)
    out = bridge(v, t)
    assert out.shape == (4, 8)
    assert out.min() >= -torch.pi and out.max() <= torch.pi


def test_quantum_multimodal_fusion_core_forward_and_backward():
    model = QuantumMultimodalFusionCore(visual_dim=512, text_dim=384, n_qubits=8, n_layers=3, num_classes=2)
    v = torch.randn(3, 512)
    t = torch.randn(3, 384)
    logits = model(v, t)
    assert logits.shape == (3, 2)

    loss = logits.sum()
    loss.backward()
    assert model.bridge.projection[0].weight.grad is not None


def test_classical_fusion_baseline():
    baseline = ClassicalMultimodalFusionBaseline(visual_dim=512, text_dim=384, latent_dim=8, num_classes=2)
    v = torch.randn(3, 512)
    t = torch.randn(3, 384)
    logits = baseline(v, t)
    assert logits.shape == (3, 2)
