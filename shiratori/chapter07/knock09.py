from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAIN_PATH = "chapter07/data/SST-2/train.tsv"
DEV_PATH = "chapter07/data/SST-2/dev.tsv"


def text2bow(text):
    return dict(Counter(text.split()))


def load_dataset(path):
    df = pd.read_csv(path, sep="\t")

    features = [text2bow(sentence) for sentence in df["sentence"]]

    labels = df["label"]

    return features, labels


def main():
    X_train_dict, y_train = load_dataset(TRAIN_PATH)
    X_dev_dict, y_dev = load_dataset(DEV_PATH)

    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform(X_train_dict)
    X_dev = vectorizer.transform(X_dev_dict)

    c_values = [0.001, 0.01, 0.1, 1, 10, 100]

    accuracies = []

    for c in c_values:
        model = LogisticRegression(C=c, max_iter=1000)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_dev)

        acc = accuracy_score(y_dev, y_pred)

        accuracies.append(acc)

        print(f"C={c}: {acc:.4f}")

    plt.plot(c_values, accuracies, marker="o")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.title("Regularization Parameter vs Accuracy")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
