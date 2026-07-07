"""73. モデルの学習

問題72のモデルを訓練セット上で学習する。
- 学習中は単語埋め込み行列の値を固定する (freeze=True)。
- 損失値を表示して学習の進捗をモニタリングする。

ここではミニバッチを使わず 1 事例ずつ学習する (ミニバッチ化は問題76)。
学習後、重みを knock73_model.pt に保存し、問題74で読み込めるようにする。
"""

import random

import torch
import torch.nn as nn

from knock70 import load_embeddings
from knock71 import DATA_DIR, load_dataset
from knock72 import BoWClassifier

MODEL_PATH = "knock73_model.pt"
N_EPOCHS = 10
LR = 0.01
SEED = 42


def train(model, train_data, n_epochs=N_EPOCHS, lr=LR, log_every=2000):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        random.shuffle(train_data)
        model.train()
        running_loss = 0.0
        for i, ex in enumerate(train_data, start=1):
            optimizer.zero_grad()
            logit = model(ex["input_ids"])          # (1, 1)
            loss = criterion(logit, ex["label"].unsqueeze(0))  # label: (1,) -> (1,1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % log_every == 0:
                avg = running_loss / log_every
                print(f"  epoch {epoch} step {i}/{len(train_data)}  loss={avg:.4f}")
                running_loss = 0.0

        # エポック全体の平均損失と訓練正解率
        train_loss, train_acc = evaluate(model, train_data, criterion)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

    return model


@torch.no_grad()
def evaluate(model, data, criterion=None):
    """1 事例ずつ評価し (loss, accuracy) を返す。"""
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss, correct = 0.0, 0
    for ex in data:
        logit = model(ex["input_ids"])
        target = ex["label"].unsqueeze(0)
        total_loss += criterion(logit, target).item()
        pred = (torch.sigmoid(logit) >= 0.5).float()
        correct += (pred.view(-1) == ex["label"]).item()
    return total_loss / len(data), correct / len(data)


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    print(f"train size: {len(train_data)}")

    model = BoWClassifier(embedding_matrix, freeze=True)
    train(model, train_data)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nsaved trained model to {MODEL_PATH}")
