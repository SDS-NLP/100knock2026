import pickle
from collections import Counter

import pandas as pd
from sklearn.metrics import classification_report

MODEL_PATH = "chapter07/data/logistic_regression_sst2.pkl"
VECTORIZER_PATH = "chapter07/data/sst2_vectorizer.pkl"
DEV_PATH = "chapter07/data/SST-2/dev.tsv"
TRAIN_PATH = "chapter07/data/SST-2/train.tsv"


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_score(df, model, vectorizer):
    y_valid = df["label"]

    features = [dict(Counter(sentence.split())) for sentence in df["sentence"]]

    X_valid = vectorizer.transform(features)

    # 予測
    y_pred = model.predict(X_valid)

    print(classification_report(y_valid, y_pred))


def main():
    model = load_data(MODEL_PATH)
    vectorizer = load_data(VECTORIZER_PATH)

    dev_df = pd.read_csv(DEV_PATH, sep="\t")
    train_df = pd.read_csv(TRAIN_PATH, sep="\t")

    get_score(dev_df, model, vectorizer)
    get_score(train_df, model, vectorizer)


if __name__ == "__main__":
    main()
