"""67. 精度の計測
学習したロジスティック回帰モデルの正解率、適合率、再現率、
F1スコアを、学習データおよび検証データ上で計測する。
"""

from __future__ import annotations

import pickle

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from knock61 import DATA_DIR, load_dataset


MODEL_PATH = DATA_DIR / "logistic_regression_sst2.pkl"
VECTORIZER_PATH = DATA_DIR / "sst2_vectorizer.pkl"


def load_artifacts():
    """Load the trained model and vectorizer."""
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def evaluate(model, vectorizer, data, split_name: str) -> None:
    x = vectorizer.transform([d["feature"] for d in data])
    y_true = [int(d["label"]) for d in data]
    y_pred = model.predict(x)

    print(f"{split_name}:")
    print(f"  accuracy : {accuracy_score(y_true, y_pred):.6f}")
    print(f"  precision: {precision_score(y_true, y_pred):.6f}")
    print(f"  recall   : {recall_score(y_true, y_pred):.6f}")
    print(f"  f1       : {f1_score(y_true, y_pred):.6f}")


def main() -> None:
    model, vectorizer = load_artifacts()

    train_data = load_dataset(DATA_DIR / "train.tsv")
    dev_data = load_dataset(DATA_DIR / "dev.tsv")

    evaluate(model, vectorizer, train_data, "train")
    evaluate(model, vectorizer, dev_data, "dev")


if __name__ == "__main__":
    main()
