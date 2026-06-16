import pandas as pd
from collections import Counter


def make_dataset(file_path):
    df = pd.read_csv(file_path, sep="\t")

    dataset = []

    for _, row in df.iterrows():
        text = row["sentence"]
        label = row["label"]

        feature = dict(Counter(text.split()))

        example = {"text": text, "label": str(label), "feature": feature}

        dataset.append(example)

    return dataset


train_data = make_dataset("chapter07/SST-2/train.tsv")
dev_data = make_dataset("chapter07/SST-2/dev.tsv")


print(train_data[0])
