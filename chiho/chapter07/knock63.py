"""63. 予測

学習済みロジスティック回帰モデルを用いて、検証データの先頭の事例の
ラベル（ポジティブ / ネガティブ）を予測し、正解ラベルと一致するか確認する。
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
    pred_label = int(model.predict(x_first)[0])
    true_label = int(first_example["label"])

    pred_name = "positive" if pred_label == 1 else "negative"
    true_name = "positive" if true_label == 1 else "negative"

    print(f"text: {first_example['text']}")
    print(f"predicted label: {pred_label} ({pred_name})")
    print(f"true label: {true_label} ({true_name})")
    print(f"match: {pred_label == true_label}")


if __name__ == "__main__":
    main()
