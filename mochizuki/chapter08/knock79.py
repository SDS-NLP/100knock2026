import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent


class CNNClassifier(nn.Module):
    def __init__(self, embeddings, num_filters=100, dropout=0.3, freeze=True):
        super().__init__()
        weight = torch.tensor(embeddings, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze, padding_idx=0
        )
        dim = weight.shape[1]
        self.convs = nn.ModuleList(
            [nn.Conv1d(dim, num_filters, kernel_size=k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.convs), 1)

    def forward(self, input_ids):
        mask = (input_ids != 0).unsqueeze(1)
        emb = self.embedding(input_ids).transpose(1, 2)
        feats = []
        for conv in self.convs:
            x = torch.relu(conv(emb))
            x = x.masked_fill(~mask, float("-inf"))
            x = torch.max(x, dim=2).values
            feats.append(x)
        x = self.dropout(torch.cat(feats, dim=1))
        return self.fc(x)


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

    model = CNNClassifier(
        embeddings,
        num_filters=args.num_filters,
        dropout=args.dropout,
        freeze=not args.fine_tune_embeddings,
    ).to(device)
    loader = DataLoader(
        data["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    print(model)
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

    torch.save(model.state_dict(), BASE_DIR / "model79.pt")
    print(f"saved: {BASE_DIR / 'model79.pt'}")
    print(f"dev accuracy: {accuracy(model, data['dev'], args.batch_size, device):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-filters", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fine-tune-embeddings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
