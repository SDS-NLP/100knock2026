"""76. ミニバッチ学習

問題75の collate を使ってミニバッチで学習し、開発セットの正解率を求める。
(ここでは CPU で実行。GPU 化は問題77。)

batch 対応の学習・評価関数はこのモジュールに置き、問題77以降から再利用する。
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from knock70 import load_embeddings
from knock71 import DATA_DIR, load_dataset
from knock72 import BoWClassifier
from knock75 import collate

BATCH_SIZE = 64
N_EPOCHS = 10
LR = 0.1
SEED = 42


def train_minibatch(model, train_data, dev_data=None, n_epochs=N_EPOCHS,
                    lr=LR, batch_size=BATCH_SIZE, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        collate_fn=collate)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)            # (batch, 1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

        train_loss = running_loss / len(train_data)
        msg = f"[epoch {epoch}] train_loss={train_loss:.4f}"
        if dev_data is not None:
            _, dev_acc = evaluate_batches(model, dev_data, device=device)
            msg += f" dev_acc={dev_acc:.4f}"
        print(msg)

    return model


@torch.no_grad()
def evaluate_batches(model, data, batch_size=256, device="cpu"):
    """ミニバッチで (loss, accuracy) を計算する。"""
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    total_loss, correct = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids)
        total_loss += criterion(logits, labels).item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
    return total_loss / len(data), correct / len(data)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    model = BoWClassifier(embedding_matrix, freeze=True)
    train_minibatch(model, train_data, dev_data, device="cpu")

    _, dev_acc = evaluate_batches(model, dev_data, device="cpu")
    print(f"\nfinal dev accuracy: {dev_acc:.4f}")
