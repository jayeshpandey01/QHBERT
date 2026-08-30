"""Milestone 2 — true QNLP (DisCoCat) on the synthetic SVO claim dataset.

No classical transformer anywhere in this pipeline. Grammar structure (fixed
Subject-Verb-Object, hand-built since the automatic parser is currently
broken — see src/data/qnlp_synthetic_claims.py) becomes the circuit topology
directly, and the circuit's own measurement outcome is the class prediction.

Run: .venv/Scripts/python.exe milestones/02_qnlp_synthetic_svo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from lambeq import IQPAnsatz, PennyLaneModel

from src.data.qnlp_diagrams import N, S, svo_diagram
from src.data.qnlp_synthetic_claims import generate_claims, train_valid_test_split

N_LAYERS = 1
EPOCHS = 100
LR = 0.1


def build_circuits(rows):
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=N_LAYERS)
    diagrams = [svo_diagram(r["subject"], r["verb"], r["object"]) for r in rows]
    return [ansatz(d) for d in diagrams]


def main():
    claims = generate_claims()
    splits = train_valid_test_split(claims)
    n_train, n_valid = len(splits["train"]), len(splits["valid"])
    print(f"train: {n_train}, valid: {n_valid}, test: {len(splits['test'])}")

    # Built from ALL rows together so every word's parameters are indexed
    # once and shared across splits — a word seen only in train still needs
    # to exist in the same parameter space the test circuits reference.
    all_rows = splits["train"] + splits["valid"] + splits["test"]
    all_circuits = build_circuits(all_rows)

    model = PennyLaneModel.from_diagrams(all_circuits, probabilities=True, normalize=True)
    model.initialise_weights()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters (one per word-qubit-rotation): {n_params}")

    train_circuits = all_circuits[:n_train]
    valid_circuits = all_circuits[n_train:n_train + n_valid]
    test_circuits = all_circuits[n_train + n_valid:]

    train_labels = torch.tensor([r["label"] for r in splits["train"]])
    valid_labels = torch.tensor([r["label"] for r in splits["valid"]])
    test_labels = torch.tensor([r["label"] for r in splits["test"]])

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.NLLLoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        probs = model(train_circuits)
        loss = criterion(torch.log(probs + 1e-9), train_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_probs = model(valid_circuits)
                val_acc = (val_probs.argmax(dim=1) == valid_labels).float().mean().item()
            print(f"Epoch {epoch + 1}/{EPOCHS} — loss: {loss.item():.4f}, val_acc: {val_acc:.4f}")

    model.eval()
    with torch.no_grad():
        test_probs = model(test_circuits)
        test_acc = (test_probs.argmax(dim=1) == test_labels).float().mean().item()
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")

    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "experiments", "qnlp_synthetic_svo_model.lt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save(checkpoint_path)
    print(f"Saved trained model to {checkpoint_path}")


if __name__ == "__main__":
    main()
