"""79. アーキテクチャの変更

問題78では、平均単語埋め込みを一つの線形層に入力して分類した。
ここでは分類器を多層パーセプトロン (MLP) に変更し、非線形な分類境界を
学習できるようにする。単語埋め込みも問題78と同様にファインチューニングする。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock73 import set_seed
from knock75 import collate
from knock77 import get_device


EPOCHS = 10
BATCH_SIZE = 64
EMBEDDING_LEARNING_RATE = 1e-3
CLASSIFIER_LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
DROPOUT = 0.3


class MLPSentenceClassifier(nn.Module):
    """平均単語埋め込みを多層ニューラルネットワークで分類するモデル。"""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        # sparse=True にすると、バッチ内で使った単語だけに勾配が作られる。
        # Google News の巨大な埋め込み行列をファインチューニングする際に
        # 勾配用メモリを節約できる。
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0,
            sparse=True,
        )

        embedding_dim = embedding_matrix.size(1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """PADを除いて単語埋め込みを平均し、文ベクトルを作る。"""
        embeddings = self.embedding(input_ids)
        valid_mask = (input_ids != 0).unsqueeze(-1)
        summed = (embeddings * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """各文がポジティブであることを表すlogitを返す。"""
        sentence_vectors = self.encode(input_ids)
        return self.classifier(sentence_vectors).squeeze(-1)

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """ポジティブである確率を返す。"""
        return torch.sigmoid(self(input_ids))


def train_model(
    model: MLPSentenceClassifier,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
) -> list[float]:
    """単語埋め込みとMLPをミニバッチ学習する。"""
    criterion = nn.BCEWithLogitsLoss()

    # sparseな埋め込みにAdamは使えないため、埋め込みはSGDで更新する。
    # パラメータ数が小さいMLPにはAdamを使い、効率よく学習させる。
    embedding_optimizer = torch.optim.SGD(
        model.embedding.parameters(), lr=EMBEDDING_LEARNING_RATE
    )
    classifier_optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=CLASSIFIER_LEARNING_RATE
    )

    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].view(-1).to(device)

            embedding_optimizer.zero_grad()
            classifier_optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()

            embedding_optimizer.step()
            classifier_optimizer.step()

            total_loss += float(loss.item()) * input_ids.size(0)

        average_loss = total_loss / len(train_loader.dataset)
        loss_history.append(average_loss)
        print(f"epoch {epoch:02d} loss={average_loss:.6f}")

    return loss_history


def compute_accuracy(
    model: MLPSentenceClassifier,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    """データセットに対する正解率をミニバッチ単位で求める。"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].view(-1).to(device)

            predictions = (model.predict_proba(input_ids) >= 0.5).float()
            correct += int((predictions == labels).sum().item())
            total += labels.size(0)

    return correct / total


def main() -> None:
    set_seed()
    device = get_device()
    print(f"device: {device}")

    embedding_matrix, token_to_id, _ = load_embedding_matrix()
    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)

    model = MLPSentenceClassifier(embedding_matrix).to(device)
    print(model)
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

    train_model(model, train_loader, device)

    train_accuracy = compute_accuracy(model, train_loader, device)
    dev_accuracy = compute_accuracy(model, dev_loader, device)
    print(f"train accuracy: {train_accuracy:.6f}")
    print(f"dev accuracy: {dev_accuracy:.6f}")


if __name__ == "__main__":
    main()
