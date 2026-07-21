import csv
import pickle
import torch

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"

token_to_id_path = "token_to_id.pkl"

def load_token_to_id(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def text_to_ids(text, token_to_id):
    ids = []

    for token in text.split():
        if token in token_to_id:
            ids.append(token_to_id[token])

    return ids

def load_dataset(path, token_to_id):
    dataset = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = float(row["label"])

            input_ids = text_to_ids(text, token_to_id)

            if len(input_ids) == 0:
                continue

            example = {
                "text": text,
                "label": torch.tensor([label], dtype=torch.float32),
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
            }

            dataset.append(example)

    return dataset

token_to_id = load_token_to_id(token_to_id_path)

train_dataset = load_dataset(train_path, token_to_id)
dev_dataset = load_dataset(dev_path, token_to_id)

print(train_dataset[0])

torch.save(train_dataset, "sst_train.pt")
torch.save(dev_dataset, "sst_dev.pt")