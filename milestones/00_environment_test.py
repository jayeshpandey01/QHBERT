"""Milestone 0 — verify the local quantum dev environment works.

Run: .venv/Scripts/python.exe milestones/00_environment_test.py
"""

import pennylane as qml

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def bell_pair_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))


if __name__ == "__main__":
    result = bell_pair_circuit()
    print(f"Quantum OK: <Z0> = {result:.4f} (expect ~0.0 for a Bell pair)")

    import torch
    print(f"Torch OK: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.models import QHBERTCore

    model = QHBERTCore()
    dummy = torch.randn(2, 768)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"QHBERTCore forward OK: output shape {tuple(out.shape)}, trainable params = {n_params}")
