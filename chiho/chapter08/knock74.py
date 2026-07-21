"""74. モデルの評価

問題73で学習したモデルを開発セットで評価し、正解率を求める。
"""

from __future__ import annotations

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets
from knock72 import SentenceLogisticRegression
from knock73 import compute_accuracy, set_seed, train_model


def main() -> None:
    set_seed()
    embedding_matrix, token_to_id, _ = load_embedding_matrix()
    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)
    model = SentenceLogisticRegression(embedding_matrix)

    train_model(model, train_dataset)

    dev_accuracy = compute_accuracy(model, dev_dataset)
    print(f"dev accuracy: {dev_accuracy:.6f}")


if __name__ == "__main__":
    main()
