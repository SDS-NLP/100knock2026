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
    # データ読み込み
    train_data = make_dataset("chapter07/SST-2/train.tsv")
    dev_data = make_dataset("chapter07/SST-2/dev.tsv")

    # 特徴量行列作成
    vectorizer = DictVectorizer()

    X_train = vectorizer.fit_transform([data["feature"] for data in train_data])

    X_dev = vectorizer.transform([data["feature"] for data in dev_data])

    y_train = [data["label"] for data in train_data]

    # モデル学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 検証データ先頭事例
    x = X_dev[0]

    # 条件付き確率
    probabilities = model.predict_proba(x)[0]

    print("Text:")
    print(dev_data[0]["text"])
    print()

    for label, prob in zip(model.classes_, probabilities):
        sentiment = "negative" if label == 0 else "positive"
        print(f"P({sentiment} | x) = {prob:.6f}")


if __name__ == "__main__":
    main()
