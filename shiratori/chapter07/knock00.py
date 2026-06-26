import pandas as pd

# polars


def count_pos_neg(file_path):
    positive = 0
    negative = 0

    df = pd.read_csv(file_path, sep="\t")

    for i in range(len(df)):
        if df.iloc[i]["label"] == 0:
            negative += 1
        else:
            positive += 1

    print(f"{file_path}: positive {positive}, negative {negative}")


if __name__ == "__main__":
    file1 = "data/SST-2/train.tsv"
    file2 = "data/SST-2/dev.tsv"

    count_pos_neg(file1)
    count_pos_neg(file2)
