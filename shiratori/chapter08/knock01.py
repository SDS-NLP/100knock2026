import pickle

import pandas as pd
import torch


with open("data/sst2_word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)


def load_dataset(path):
    data = pd.read_csv(path, sep="\t")

    dataset = []

    for _, row in data.iterrows():
        text = row["sentence"]
        label = row["label"]

        tokens = text.split()

        ids = []
        for token in tokens:
            if token in word_to_id:
                ids.append(word_to_id[token])
        # 全部未知なら飛ばす
        if len(ids) == 0:
            continue

        dataset.append(
            {
                "text": text,
                "label": torch.tensor([label], dtype=torch.float),
                "input_ids": torch.tensor(ids, dtype=torch.long),
            }
        )

    return dataset


def main():
    train_dataset = load_dataset("data/SST-2/train.tsv")
    dev_dataset = load_dataset("data/SST-2/dev.tsv")

    print(train_dataset[0])
    print(f"train size: {len(train_dataset)}")
    print(f"dev size: {len(dev_dataset)}")

    with open("data/sst2_train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)

    with open("data/sst2_dev_dataset.pkl", "wb") as f:
        pickle.dump(dev_dataset, f)


if __name__ == "__main__":
    main()
