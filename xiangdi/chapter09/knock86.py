import csv
import torch
from transformers import AutoTokenizer

train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_dataset(path):
    dataset = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            text = row["sentence"]
            label = int(row["label"])
            input_ids = tokenizer.encode(text)

            dataset.append({
                "text": text,
                "label": label,
                "input_ids": input_ids,
            })

    return dataset


def collate(batch):
    max_length = max(len(example["input_ids"]) for example in batch)

    input_ids = []
    attention_masks = []
    labels = []

    for example in batch:
        ids = example["input_ids"]
        padding_length = max_length - len(ids)

        input_ids.append(ids + [tokenizer.pad_token_id] * padding_length)
        attention_masks.append([1] * len(ids) + [0] * padding_length)
        labels.append(example["label"])

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_masks),
        "labels": torch.tensor(labels),
    }


train_dataset = load_dataset(train_path)
batch = collate(train_dataset[:4])

print(batch["input_ids"])
