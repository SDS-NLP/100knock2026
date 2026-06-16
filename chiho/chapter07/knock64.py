"""64. 条件付き確率

学習したロジスティック回帰モデルを用いて、
検証データの先頭の事例を各ラベルに分類するときの条件付き確率を求める。
"""

from __future__ import annotations

import pickle

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

    first_example = dev_data[0]
    x_first = vectorizer.transform([first_example["feature"]])
    probabilities = model.predict_proba(x_first)[0]

    print(f"text: {first_example['text']}")
    print("conditional probabilities:")
    for label, probability in zip(model.classes_, probabilities):
        label_name = "positive" if int(label) == 1 else "negative"
        print(f"  P(label={label} | x) = {probability:.6f} ({label_name})")


if __name__ == "__main__":
    main()
