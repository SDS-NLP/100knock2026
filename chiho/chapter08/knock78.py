"""78. 単語埋め込みのファインチューニング

問題77では、事前学習済み単語埋め込みを固定して線形層だけを学習した。
ここでは単語埋め込みのパラメータも同時に更新する。（バッチサイズ: 8）
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock73 import set_seed
from knock75 import collate


EPOCHS = 10
LEARNING_RATE = 1e-2
BATCH_SIZE = 8


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTuningSentenceLogisticRegression(nn.Module):
    """
    単語埋め込みの平均ベクトルで文を表し、2値分類するモデル
    問題72の SentenceLogisticRegression とほぼ同じ
    nn.Embedding.from_pretrained(..., freeze=False) にしている点が違う。
    freeze=False にすることで学習中に単語埋め込み行列も更新
    """

    def __init__(self, embedding_matrix: torch.Tensor) -> None:
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            # 埋め込み層をつくる
            embedding_matrix, # 初期値
            freeze=False,  # ファインチューニング！単語ベクトルも学習中に更新される
            padding_idx=0,  # 0番は <PAD> 
            sparse=True,  # 巨大な埋め込み行列でも、使った単語だけ勾配を持つようにする
        )
        self.linear = nn.Linear(embedding_matrix.size(1), 1)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        単語ID列を文ベクトルにする
        長さを揃えるためのパディングを考慮して平均ベクトルをとる
        """
        embeddings = self.embedding(input_ids)

        # input_ids != 0 の位置だけ True になるマスクを作る。
        # unsqueeze(-1) で 「バッチサイズ × 文の長さ」に次元を一つ追加して300次元全体にかけやすくする
        valid_mask = (input_ids != 0).unsqueeze(-1)

        # PAD 部分のベクトルを 0 にしてから、トークン方向に合計する。
        summed = (embeddings * valid_mask).sum(dim=1)

        # PAD を除いた実際の単語数。0除算を避けるため clamp(min=1) しておく
        lengths = valid_mask.sum(dim=1).clamp(min=1)

        return summed / lengths

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """logits を返す"""
        sentence_vectors = self.encode(input_ids)
        logits = self.linear(sentence_vectors)
        return logits.squeeze(-1)

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """logits に sigmoid をかけ、ポジティブである確率に変換する"""
        return torch.sigmoid(self(input_ids))


def train_finetuning_model(
    model: FineTuningSentenceLogisticRegression,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list[float]:
    
    """単語埋め込みと線形層を同時に学習
    - 問題77: optimizer = SGD(model.linear.parameters(), ...)
      （線形層だけ更新）
    - 問題78: optimizer = SGD(model.parameters(), ...)
      （埋め込み層と線形層の両方更新）
    """
    criterion = nn.BCEWithLogitsLoss() # 損失関数　logits → sigmoid → binary cross entropyまで
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # model.parameters...これでモデル全体が更新対象になってる

    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].view(-1).to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * input_ids.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        loss_history.append(average_loss)
        print(f"epoch {epoch:02d} loss={average_loss:.6f}")

    return loss_history


def compute_accuracy(
    model: FineTuningSentenceLogisticRegression,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """開発データで正解率確認"""
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
    print(f"embedding dim: {embedding_matrix.size(1)}")

    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)

    # モデル全体を GPU に移す
    # freeze=False の埋め込み層も一緒に GPU に移る
    model = FineTuningSentenceLogisticRegression(embedding_matrix).to(device)
    print(f"embedding trainable: {model.embedding.weight.requires_grad}")

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

    train_finetuning_model(model, train_loader, device)

    dev_accuracy = compute_accuracy(model, dev_loader, device)
    print(f"dev accuracy: {dev_accuracy:.6f}")


if __name__ == "__main__":
    main()
