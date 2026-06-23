import pickle
from collections import Counter

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


DATA_DIR = "chapter07/SST-2"

MODEL_PATH = "logistic_regression_sst2.pkl"
VECTORIZER_PATH = "sst2_vectorizer.pkl"


def load_dataset(path):
    df = pd.read_csv(path, sep="\t")

    data = []
    for _, row in df.iterrows():
        feature = dict(Counter(row["sentence"].split()))
        data.append((feature, int(row["label"])))

    return data


def vectorize(train_data, dev_data):
    vectorizer = DictVectorizer(sparse=False)

    X_train = vectorizer.fit_transform([x for x, _ in train_data])
    X_dev = vectorizer.transform([x for x, _ in dev_data])

    return X_train, X_dev, vectorizer


def train(X_train, y_train):
    model = LogisticRegression(solver="liblinear", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    train_data = load_dataset(f"{DATA_DIR}/train.tsv")
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv")

    X_train, X_dev, vectorizer = vectorize(train_data, dev_data)

    y_train = [y for _, y in train_data]
    y_dev = [y for _, y in dev_data]

    model = train(X_train, y_train)

    pred = model.predict(X_dev)
    acc = accuracy_score(y_dev, pred)

    print("dev accuracy:", acc)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)


if __name__ == "__main__":
    main()
