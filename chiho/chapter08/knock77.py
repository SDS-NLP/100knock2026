"""77. GPU上での学習

問題76のミニバッチ学習を GPU 上で実行する。
単語埋め込みの次元数は Google News ベクトルの 300 次元、バッチサイズは 8。
学習後に開発セット (dev) の正解率を求める。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock72 import SentenceLogisticRegression
from knock73 import set_seed
from knock75 import collate


EPOCHS = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 8


def get_device() -> torch.device:
    """学習に使うデバイスを決める。

    cuda が使える環境では NVIDIA GPU を使う。
    cuda がない場合でもコードを読んだり動かしたりしやすいように CPU に戻す。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_on_gpu(
    model: SentenceLogisticRegression,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list[float]:
    """GPU 上でミニバッチ学習を行う。

    DataLoader から出てくるテンソルは最初 CPU 上にある。
    そのため、各バッチを使う直前に .to(device) で GPU へ移す。
    """
    criterion = nn.BCEWithLogitsLoss()

    # 埋め込み層は freeze=True なので更新対象は線形層だけ。
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=learning_rate)

    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # モデルと入力データは同じデバイス上にないと計算できない。
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].view(-1).to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # loss.item() はバッチ平均なので、バッチサイズを掛けて合計損失に戻す。
            total_loss += float(loss.item()) * input_ids.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        loss_history.append(average_loss)
        print(f"epoch {epoch:02d} loss={average_loss:.6f}")

    return loss_history


def compute_accuracy_on_gpu(
    model: SentenceLogisticRegression,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """GPU 上で予測し、正解率を計算する。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].view(-1).to(device)

            probabilities = model.predict_proba(input_ids)
            predictions = (probabilities >= 0.5).float()

            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    return correct / total


def main() -> None:
    set_seed()
    device = get_device()
    print(f"device: {device}")

    embedding_matrix, token_to_id, _ = load_embedding_matrix()

    # 問題の指定どおり、Google News ベクトルの埋め込み次元数は 300。
    print(f"embedding dim: {embedding_matrix.size(1)}")

    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)

    # モデルを作ってから .to(device) で GPU へ移す。
    # nn.Embedding.from_pretrained で作った埋め込み行列も、ここで一緒に移動される。
    model = SentenceLogisticRegression(embedding_matrix).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    train_model_on_gpu(model, train_loader, device)

    dev_accuracy = compute_accuracy_on_gpu(model, dev_loader, device)
    print(f"dev accuracy: {dev_accuracy:.6f}")


if __name__ == "__main__":
    main()
