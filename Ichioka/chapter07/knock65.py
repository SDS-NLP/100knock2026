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
    # 1. データ読み込み
    train_data = load_dataset_as_bow(train_path)
    dev_data = load_dataset_as_bow(dev_path)

    # 2. 特徴量とラベルを取り出す
    X_train_dict = [example["feature"] for example in train_data]
    y_train = [example["label"] for example in train_data]

    X_dev_dict = [example["feature"] for example in dev_data]
    y_dev = [example["label"] for example in dev_data]

    # 3. BoW辞書をベクトル化
    vectorizer = DictVectorizer()
    X_train = vectorizer.fit_transform(X_train_dict)
    X_dev = vectorizer.transform(X_dev_dict)

    # 4. ロジスティック回帰モデルを学習
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. 検証データで精度確認
    dev_accuracy = model.score(X_dev, y_dev)
    print(f"dev accuracy: {dev_accuracy:.4f}")

    # 6. 任意のテキストを予測
    input_text = "the worst movie I 've ever seen"

    input_feature = text2bow(input_text)
    X_input = vectorizer.transform([input_feature])

    pred_label = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]

    print()
    print("入力文:", input_text)
    print("予測ラベル:", pred_label)

    for label, prob in zip(model.classes_, pred_proba):
        print(f"P(y={label} | x) = {prob:.4f}")

    if pred_label == "1":
        print("予測結果: positive")
    else:
        print("予測結果: negative")


if __name__ == "__main__":
    main()