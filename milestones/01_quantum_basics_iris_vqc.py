"""Milestone 1 — first working Variational Quantum Classifier, on Iris.

sklearn ships the Iris dataset built in, so this needs no downloads.
Binary task: class 0 (setosa) vs. classes 1&2 (versicolor/virginica).

Run: .venv/Scripts/python.exe milestones/01_quantum_basics_iris_vqc.py
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def vqc(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


class IrisVQC(nn.Module):
    def __init__(self, n_layers: int = 3):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qlayer = qml.qnn.TorchLayer(vqc, weight_shapes)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.qlayer(x) + self.bias


def main():
    X, y = load_iris(return_X_y=True)
    y_binary = (y != 0).astype(np.float32)  # setosa=0 vs rest=1

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32) * 2 - 1  # map {0,1} -> {-1,+1}
    y_test_labels = torch.tensor(y_test, dtype=torch.float32)

    model = IrisVQC()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    epochs = 30
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = torch.stack([model(x) for x in X_train])
        loss = loss_fn(preds, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} — loss: {loss.item():.4f}")

    with torch.no_grad():
        test_preds = torch.stack([model(x) for x in X_test])
        predicted_labels = (test_preds > 0).float()
        accuracy = (predicted_labels == y_test_labels).float().mean().item()

    print(f"\nTest accuracy: {accuracy * 100:.2f}%  (target: >85%)")


if __name__ == "__main__":
    main()
