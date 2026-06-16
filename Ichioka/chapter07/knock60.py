import pandas as pd

train_path = "SST-2/SST-2/train.tsv"
dev_path = "SST-2/SST-2/dev.tsv"

def count_labels(file_path: str):
    # TSVファイルを読み込み
    df = pd.read_csv(file_path, sep="\t")

    # label列の値をカウント
    counts = df["label"].value_counts().sort_index()

    positive_count = counts.get(1, 0)
    negative_count = counts.get(0, 0)

    print(f"== {file_path} ==")
    print(f"Positive (1): {positive_count}")
    print(f"Negative (0): {negative_count}")
    print(f"Total: {positive_count + negative_count}")
    print()


def main():
    count_labels(dev_path)
    count_labels(train_path)


if __name__ == "__main__":
    main()