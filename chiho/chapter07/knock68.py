"""68. 特徴量の重みの確認

学習したロジスティック回帰モデルの中で、重みの高い特徴量トップ20と、
重みの低い特徴量トップ20を確認する。
"""

from __future__ import annotations

import pickle

from knock61 import DATA_DIR


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

    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    ranked_features = sorted(zip(feature_names, weights), key=lambda x: x[1], reverse=True)

    print("top 20 positive features:")
    for feature, weight in ranked_features[:20]:
        print(f"  {feature}: {weight:.6f}")

    print("top 20 negative features:")
    for feature, weight in ranked_features[-20:][::-1]:
        print(f"  {feature}: {weight:.6f}")


if __name__ == "__main__":
    main()
