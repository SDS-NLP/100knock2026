import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"

def text_to_feature(text):
    return dict(Counter(text.split()))

def load_sst2_as_dict_list(path):
    df = pd.read_csv(path, sep="\t")

    data = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = str(row["label"])

        data.append({
            "text": text,
            "label": label,
            "feature": text_to_feature(text)
        })

    return data

train_data = load_sst2_as_dict_list(train_path)
dev_data = load_sst2_as_dict_list(dev_path)

X_train_dict = [example["feature"] for example in train_data]
y_train = [int(example["label"]) for example in train_data]

X_dev_dict = [example["feature"] for example in dev_data]
y_dev = [int(example["label"]) for example in dev_data]

vectorizer = DictVectorizer()
X_train = vectorizer.fit_transform(X_train_dict)
X_dev = vectorizer.transform(X_dev_dict)

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

results = []

for C in C_values:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    y_dev_pred = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_dev_pred)

    results.append({
        "C": C,
        "accuracy": accuracy
    })

results_df = pd.DataFrame(results)
print(results_df)

plt.figure()
plt.plot(results_df["C"], results_df["accuracy"], marker="o")
plt.xscale("log")
plt.xlabel("Regularization parameter C")
plt.ylabel("Accuracy on dev data")
plt.title("Accuracy for different regularization parameters")
plt.grid(True)
plt.show()