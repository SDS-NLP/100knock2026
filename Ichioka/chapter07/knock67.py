import pandas as pd
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


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

    # ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 学習データ上で予測
    y_train_pred = model.predict(X_train)

    # 検証データ上で予測
    y_dev_pred = model.predict(X_dev)

    # 学習データ上での評価
    print("Train")
    print(classification_report(
        y_train,
        y_train_pred,
        labels=["0", "1"],
        target_names=["negative", "positive"],
    ))

    # 検証データ上での評価
    print("Dev")
    print(classification_report(
        y_dev,
        y_dev_pred,
        labels=["0", "1"],
        target_names=["negative", "positive"],
    ))


if __name__ == "__main__":
    main()