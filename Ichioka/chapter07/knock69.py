import pandas as pd
from collections import Counter

import matplotlib.pyplot as plt

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


train_path = "SST-2/SST-2/train.tsv"
dev_path = "SST-2/SST-2/dev.tsv"


def text2bow(text: str):
    """スペース区切りのトークンに基づいてBoW特徴ベクトルを作る."""
    tokens = text.split()
    return dict(Counter(tokens))


def load_dataset_as_bow(file_path: str):
    """各事例を辞書オブジェクトに変換する."""
    df = pd.read_csv(file_path, sep="\t")

    examples = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = str(row["label"])

        example = {
            "text": text,
            "label": label,
            "feature": text2bow(text),
        }

        examples.append(example)

    return examples


def main():
    # データ読み込み
    train_data = load_dataset_as_bow(train_path)
    dev_data = load_dataset_as_bow(dev_path)

    # 特徴量とラベル
    X_train_dict = [example["feature"] for example in train_data]
    y_train = [example["label"] for example in train_data]

    X_dev_dict = [example["feature"] for example in dev_data]
    y_dev = [example["label"] for example in dev_data]

    # BoW辞書をベクトル化
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train_dict)
    X_dev = vectorizer.transform(X_dev_dict)

    # 正則化パラメータ C を変化させる
    # 注意点として、C は 正則化係数そのものではなく、正則化の強さの逆数
    C_values = [
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1.0,
        5.0,
        10.0,
        50.0,
        100.0,
    ]

    dev_accuracies = []

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
        )

        model.fit(X_train, y_train)

        y_dev_pred = model.predict(X_dev)
        dev_accuracy = accuracy_score(y_dev, y_dev_pred)

        dev_accuracies.append(dev_accuracy)

    # グラフを作成
    plt.figure(figsize=(10, 10))
    plt.plot(C_values, dev_accuracies, marker="o")

    plt.xscale("log")
    plt.xlabel("Regularization parameter C")
    plt.ylabel("Dev accuracy")
    plt.title("Dev accuracy for different regularization strengths")

    plt.grid(True)
    plt.tight_layout()

    # 画像として保存
    plt.savefig("regularization_accuracy.png")


if __name__ == "__main__":
    main()