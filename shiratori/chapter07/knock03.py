from collections import Counter
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def make_dataset(file_path):
    df = pd.read_csv(file_path, sep="\t")

    dataset = []

    for _, row in df.iterrows():
        feature = dict(Counter(row["sentence"].split()))

        dataset.append({"text": row["sentence"], "label": row["label"], "feature": feature})

    return dataset


def main():
    train_data = make_dataset("data/SST-2/train.tsv")
    dev_data = make_dataset("datas/SST-2/dev.tsv")

    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform([data["feature"] for data in train_data])

    X_dev = vectorizer.transform([data["feature"] for data in dev_data])

    y_train = [data["label"] for data in train_data]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred_label = model.predict(X_dev[0])[0]
    true_label = dev_data[0]["label"]

    label_name = {0: "negative", 1: "positive"}

    print("Text:")
    print(dev_data[0]["text"])

    print("\nPredicted:", label_name[pred_label])
    print("True:", label_name[true_label])

    if pred_label == true_label:
        print("Result: Correct")
    else:
        print("Result: Incorrect")


if __name__ == "__main__":
    main()
