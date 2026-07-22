import os

import torch

from knock80 import load_tokenizer
from knock85 import load_dataset


def make_batch(texts, labels, tokenizer):
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    batch["labels"] = torch.tensor(labels)
    return batch


def main():
    tokenizer = load_tokenizer()
    data_path = os.path.join(os.path.dirname(__file__), "data", "SST-2", "train.tsv")
    _, train_texts, train_labels, _ = load_dataset(
        data_path, tokenizer
    )
    batch = make_batch(train_texts[:4], train_labels[:4], tokenizer)

    for key, value in batch.items():
        print(f"{key}:\n{value}")


if __name__ == "__main__":
    main()
