import pickle
import pandas as pd

MODEL_PATH = "chapter07/data/logistic_regression_sst2.pkl"
VECTORIZER_PATH = "chapter07/data/sst2_vectorizer.pkl"


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_weights(model, vectorizer):
    coef_df = pd.DataFrame({"feature": vectorizer.get_feature_names_out(), "weight": model.coef_[0]})

    print("=== 重みの高い特徴量 Top 20 ===")
    print(coef_df.sort_values("weight", ascending=False).head(20))

    print()

    print("=== 重みの低い特徴量 Top 20 ===")
    print(coef_df.sort_values("weight", ascending=True).head(20))


def main():
    model = load_data(MODEL_PATH)
    vectorizer = load_data(VECTORIZER_PATH)

    get_feature_weights(model, vectorizer)


if __name__ == "__main__":
    main()
