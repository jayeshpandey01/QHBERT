"""Train QHBERTCore on cached DistilBERT embeddings (produced by
kaggle/extract_bert_embeddings.py). Runs fine on CPU: the frozen BERT pass
already happened on Kaggle, so this only trains the ~50K-parameter
bridge+quantum+head stack.

Usage:
  .venv/Scripts/python.exe -m src.training.train --embeddings path/to/liar_embeddings.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models import QHBERTCore


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for embeddings, labels in loader:
            logits = model(embeddings)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to cached embeddings .pt file")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-classical", type=float, default=1e-3)
    parser.add_argument("--lr-quantum", type=float, default=1e-2)
    args = parser.parse_args()

    data = torch.load(args.embeddings)
    train_ds = TensorDataset(data["train"]["embeddings"], data["train"]["labels"])
    val_key = "valid" if "valid" in data else "test"
    val_ds = TensorDataset(data[val_key]["embeddings"], data[val_key]["labels"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = QHBERTCore(bert_dim=data["train"]["embeddings"].shape[1])

    quantum_params = list(model.quantum.parameters())
    classical_params = list(model.bridge.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam([
        {"params": classical_params, "lr": args.lr_classical},
        {"params": quantum_params, "lr": args.lr_quantum},
    ])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            logits = model(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * embeddings.size(0)

        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch + 1}/{args.epochs} — train_loss: {avg_loss:.4f}, val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  New best val_acc: {val_acc:.4f} -> saved best_model.pt")

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
