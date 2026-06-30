"""65. テキストのポジネガ予測

与えられたテキストのポジネガを予測するプログラムを実装する。
例として "the worst movie I've ever seen" を与え、
ロジスティック回帰モデルの予測結果を確認する。
"""

from __future__ import annotations

import pickle

from knock61 import DATA_DIR, make_feature


MODEL_PATH = DATA_DIR / "logistic_regression_sst2.pkl"
VECTORIZER_PATH = DATA_DIR / "sst2_vectorizer.pkl"


def load_artifacts():
    """Load the trained model and vectorizer."""
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def main() -> None:
    model, vectorizer = load_artifacts()

    text = "the worst movie I've ever seen"
    feature = make_feature(text)
    x = vectorizer.transform([feature])
    pred_label = int(model.predict(x)[0])
    pred_name = "positive" if pred_label == 1 else "negative"

    print(f"text: {text}")
    print(f"predicted label: {pred_label} ({pred_name})")


if __name__ == "__main__":
    main()
