import pandas as pd
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


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

    # BoW辞書をベクトル化
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train_dict)

    # ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 特徴量名と重みを取得
    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    # 特徴量と重みをペアにする
    feature_weights = list(zip(feature_names, weights))

    # 重みが高い特徴量トップ20
    top_positive = sorted(
        feature_weights,
        key=lambda x: x[1],
        reverse=True
    )[:20]

    # 重みが低い特徴量トップ20
    top_negative = sorted(
        feature_weights,
        key=lambda x: x[1]
    )[:20]

    print("Top 20 high-weight features")
    for feature, weight in top_positive:
        print(f"{feature}\t{weight:.4f}")

    print()

    print("Top 20 low-weight features")
    for feature, weight in top_negative:
        print(f"{feature}\t{weight:.4f}")


if __name__ == "__main__":
    main()