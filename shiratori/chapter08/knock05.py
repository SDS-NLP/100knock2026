import torch
from torch.utils.data import DataLoader
import pickle

with open("data/sst2_dev_dataset.pkl", "rb") as f:
    dev = pickle.load(f)
with open("data/SST-2/train.tsv", "rb") as f:
    train = pickle.load(f)


def collate(batch):
    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)

    max_len = len(batch[0]["input_ids"])

    input_ids = []
    for sample in batch:
        ids = sample["input_ids"]
        pad_len = max_len - len(ids)

        padded = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])

        input_ids.append(padded)

    input_ids = torch.stack(input_ids)

    labels = torch.stack([sample["label"] for sample in batch])

    return {"input_ids": input_ids, "label": labels}


def main():

    train_loader = DataLoader(train, batch_size=4, shuffle=False, collate_fn=collate)
    print(train[0])

    batch = next(iter(train_loader))

    print(batch)

    # dev_loader = DataLoader(dev, batch_size=4, shuffle=False, collate_fn=collate)

    # batch = next(iter(dev_loader))
    # print(batch)


if __name__ == "__main__":
    main()
