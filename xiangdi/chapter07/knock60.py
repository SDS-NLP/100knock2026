import pandas as pd

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"

train_df = pd.read_csv(train_path, sep="\t")
dev_df = pd.read_csv(dev_path, sep="\t")

for name, df in [("train.tsv", train_df), ("dev.tsv", dev_df)]:
    counts = df["label"].value_counts().sort_index()

    print(name)
    print(f"negative (0): {counts.get(0, 0)}")
    print(f"positive (1): {counts.get(1, 0)}")