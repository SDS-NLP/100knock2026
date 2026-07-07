import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent


class BoWClassifier(nn.Module):
    def __init__(self, embeddings, freeze=False):
        super().__init__()
        weight = torch.tensor(embeddings, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze, padding_idx=0
        )
        self.fc = nn.Linear(weight.shape[1], 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        return self.fc(summed / lengths)


def collate(batch):
    batch = sorted(batch, key=lambda ex: len(ex["input_ids"]), reverse=True)
    input_ids = pad_sequence(
        [ex["input_ids"] for ex in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.stack([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "label": labels}


def accuracy(model, examples, batch_size, device):
    loader = DataLoader(
        examples, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    correct, n = 0, 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            pred = (torch.sigmoid(model(input_ids)) > 0.5).float()
            correct += (pred == labels).sum().item()
            n += labels.size(0)
    return correct / n


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(BASE_DIR / "dataset.pt")
    embeddings = np.load(BASE_DIR / "embeddings.npy")

    model = BoWClassifier(embeddings, freeze=False).to(device)
    loader = DataLoader(
        data["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        [
            {"params": model.embedding.parameters(), "lr": args.embedding_lr},
            {"params": model.fc.parameters(), "lr": args.lr},
        ]
    )

    print(f"device: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (
                (torch.sigmoid(logits) > 0.5).float() == labels
            ).sum().item()
            n += labels.size(0)

        dev_acc = accuracy(model, data["dev"], args.batch_size, device)
        print(
            f"epoch {epoch:2d}  loss={total_loss / n:.4f}  "
            f"train_acc={correct / n:.4f}  dev_acc={dev_acc:.4f}"
        )

    torch.save(model.state_dict(), BASE_DIR / "model78.pt")
    print(f"saved: {BASE_DIR / 'model78.pt'}")
    print(f"dev accuracy: {accuracy(model, data['dev'], args.batch_size, device):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--embedding-lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
