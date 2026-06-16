import pandas as pd
from collections import Counter

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
    train_data = load_dataset_as_bow(train_path)
    dev_data = load_dataset_as_bow(dev_path)

    print("学習データの事例数:", len(train_data))
    print("検証データの事例数:", len(dev_data))

    print("\n== 最初の事例 ==")
    print("text:")
    print(train_data[0]["text"])

    print("\nlabel:")
    print(train_data[0]["label"])

    print("\nfeature:")
    print(train_data[0]["feature"])


if __name__ == "__main__":
    main()