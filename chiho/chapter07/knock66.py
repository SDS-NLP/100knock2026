"""66. 混同行列の作成

学習したロジスティック回帰モデルの検証データにおける
混同行列 (confusion matrix) を求める。
"""

from __future__ import annotations

import pickle

from sklearn.metrics import confusion_matrix

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


def main() -> None:
    model, vectorizer = load_artifacts()
    dev_data = load_dataset(DATA_DIR / "dev.tsv")

    dev_x = vectorizer.transform([d["feature"] for d in dev_data])
    dev_y = [int(d["label"]) for d in dev_data]
    pred_y = model.predict(dev_x)

    cm = confusion_matrix(dev_y, pred_y, labels=model.classes_)

    print("confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
