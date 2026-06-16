from collections import Counter
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def make_dataset(file_path):
    df = pd.read_csv(file_path, sep="\t")

    dataset = []

    for _, row in df.iterrows():
        feature = dict(Counter(row["sentence"].split()))

        example = {"text": row["sentence"], "label": row["label"], "feature": feature}

        dataset.append(example)

    return dataset


train_data = make_dataset("chapter07/SST-2/train.tsv")
dev_data = make_dataset("chapter07/SST-2/dev.tsv")


vectorizer = DictVectorizer()

X_train = vectorizer.fit_transform([data["feature"] for data in train_data])

X_dev = vectorizer.transform([data["feature"] for data in dev_data])

# ラベル
y_train = [data["label"] for data in train_data]
y_dev = [data["label"] for data in dev_data]


# ロジスティック回帰モデル学習
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


y_pred = model.predict(X_dev)


accuracy = accuracy_score(y_dev, y_pred)

print("Accuracy:", accuracy)
