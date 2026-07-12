"""73. モデルの学習

問題72のロジスティック回帰モデルを訓練セットで学習する。
学習中は単語埋め込みを固定し、損失の推移を表示する。
"""

from __future__ import annotations

import random

import torch
from torch import nn

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock72 import SentenceLogisticRegression


SEED = 42
EPOCHS = 10
LEARNING_RATE = 1e-2


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    model: SentenceLogisticRegression,
    train_dataset: list[dict[str, object]],
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> list[float]:
    """Train the classifier with fixed word embeddings."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.linear.parameters(), lr=learning_rate)
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        random.shuffle(train_dataset)
        total_loss = 0.0

        for example in train_dataset:
            input_ids = example["input_ids"]
            label = example["label"].view(())

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        average_loss = total_loss / len(train_dataset)
        loss_history.append(average_loss)
        print(f"epoch {epoch:02d} loss={average_loss:.6f}")

    return loss_history


def compute_accuracy(
    model: SentenceLogisticRegression,
    dataset: list[dict[str, object]],
) -> float:
    """Compute accuracy on a dataset."""
    correct = 0

    with torch.no_grad():
        for example in dataset:
            probability = model.predict_proba(example["input_ids"])
            prediction = float(probability.item() >= 0.5)
            gold = float(example["label"].item())
            if prediction == gold:
                correct += 1

    return correct / len(dataset)


def main() -> None:
    set_seed()
    embedding_matrix, token_to_id, _ = load_embedding_matrix()
    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)
    model = SentenceLogisticRegression(embedding_matrix)

    print(f"train size: {len(train_dataset)}")
    print(f"dev size: {len(dev_dataset)}")
    print(f"embedding trainable: {model.embedding.weight.requires_grad}")

    train_model(model, train_dataset)

    train_accuracy = compute_accuracy(model, train_dataset)
    print(f"train accuracy: {train_accuracy:.6f}")


if __name__ == "__main__":
    main()
