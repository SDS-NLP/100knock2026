import pickle
from collections import Counter

import pandas as pd
from sklearn.metrics import confusion_matrix

MODEL_PATH = "chapter07/data/logistic_regression_sst2.pkl"
VECTORIZER_PATH = "chapter07/data/sst2_vectorizer.pkl"
DEV_PATH = "chapter07/data/SST-2/dev.tsv"


def load_data(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    model = load_data(MODEL_PATH)
    vectorizer = load_data(VECTORIZER_PATH)

    dev_df = pd.read_csv(DEV_PATH, sep="\t")

    y_valid = dev_df["label"]

    features = [dict(Counter(sentence.split())) for sentence in dev_df["sentence"]]

    X_valid = vectorizer.transform(features)

    y_pred = model.predict(X_valid)

    # 混同行列
    cm = confusion_matrix(y_valid, y_pred)

    print(cm)


if __name__ == "__main__":
    main()
