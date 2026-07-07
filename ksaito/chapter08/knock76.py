from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from knock71 import load_embeddings, load_sst2_as_ids
from knock72 import BagOfWordsClassifier
from knock75 import collate


def accuracy(model, examples, device="cpu", batch_size=256):
    model.eval()
    loader = DataLoader(examples, batch_size=batch_size, shuffle=False, collate_fn=collate)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            pred = (torch.sigmoid(model(input_ids)) >= 0.5).float()
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / total


def train_minibatch(model, train, dev, epochs=5, batch_size=128, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(input_ids), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        print(
            f"epoch={epoch} loss={total_loss / total_count:.4f} "
            f"dev_accuracy={accuracy(model, dev, device, batch_size):.4f}"
        )
    return model


def main():
    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    model = BagOfWordsClassifier(embedding_matrix, freeze=True)
    train_minibatch(model, train, dev, epochs=5, batch_size=128, lr=1e-3)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/knock76_bow_minibatch.pt")
    print(f"final_dev_accuracy: {accuracy(model, dev):.4f}")


if __name__ == "__main__":
    main()
