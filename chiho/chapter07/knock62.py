"""62. 学習

61で作成した BoW 特徴を使って、ロジスティック回帰モデルを学習する。
学習・検証・保存の流れが口頭でも説明しやすいように、
処理を小さな関数に分けている。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knock61 import DATA_DIR, load_dataset


MODEL_PATH = DATA_DIR / "logistic_regression_sst2.pkl"
VECTORIZER_PATH = DATA_DIR / "sst2_vectorizer.pkl"


def load_data() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load train/dev data that already contains BoW features."""
    train_data = load_dataset(DATA_DIR / "train.tsv")
    dev_data = load_dataset(DATA_DIR / "dev.tsv")
    return train_data, dev_data


def vectorize_features(
    train_data: list[dict[str, object]],
    dev_data: list[dict[str, object]],
) -> tuple[Any, Any, DictVectorizer]:
    """Convert feature dictionaries into numeric matrices."""
    # Dense matrix にして、scikit-learn の sparse index 問題を避ける。
    vectorizer = DictVectorizer(sparse=False)

    train_x = vectorizer.fit_transform([example["feature"] for example in train_data])
    dev_x = vectorizer.transform([example["feature"] for example in dev_data])

    return train_x, dev_x, vectorizer


def extract_labels(data: list[dict[str, object]]) -> list[int]:
    """Convert string labels into integers."""
    return [int(example["label"]) for example in data]


def train_model(train_x, train_y) -> LogisticRegression:
    """Train a logistic regression classifier."""
    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    model.fit(train_x, train_y)
    return model


def evaluate_model(model: LogisticRegression, dev_x, dev_y) -> float:
    """Evaluate the trained model on the development set."""
    dev_pred = model.predict(dev_x)
    accuracy = accuracy_score(dev_y, dev_pred)
    print(f"dev accuracy: {accuracy:.4f}")
    return accuracy


def save_artifacts(model: LogisticRegression, vectorizer: DictVectorizer) -> None:
    """Save the trained model and the vectorizer together."""
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)


def main() -> None:
    train_data, dev_data = load_data()
    train_x, dev_x, vectorizer = vectorize_features(train_data, dev_data)
    train_y = extract_labels(train_data)
    dev_y = extract_labels(dev_data)

    model = train_model(train_x, train_y)
    evaluate_model(model, dev_x, dev_y)
    save_artifacts(model, vectorizer)

    print("model saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {VECTORIZER_PATH}")
    print(f"n_train: {train_x.shape[0]}")
    print(f"n_features: {train_x.shape[1]}")


if __name__ == "__main__":
    main()
