import csv
from pathlib import Path

import torch

from knock70 import create_embedding_matrix, load_pretrained_embeddings


SST2_DIR = Path("../chapter07/data/SST-2")


def load_embeddings(limit=100000):
    pretrained = load_pretrained_embeddings("GoogleNews-vectors-negative300.bin.gz", limit=limit)
    return create_embedding_matrix(pretrained)


def read_sst2(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row["sentence"], int(row["label"])


def sentence_to_ids(sentence, token_to_id):
    return [token_to_id[token] for token in sentence.split() if token in token_to_id]


def load_sst2_as_ids(token_to_id):
    datasets = {}
    for split in ["train", "dev"]:
        examples = []
        for sentence, label in read_sst2(SST2_DIR / f"{split}.tsv"):
            input_ids = sentence_to_ids(sentence, token_to_id)
            if not input_ids:
                continue
            examples.append(
                {
                    "text": sentence,
                    "label": torch.tensor([float(label)], dtype=torch.float32),
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                }
            )
        datasets[split] = examples
    return datasets["train"], datasets["dev"]


def main():
    _, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    print(f"train: {len(train)} examples")
    print(f"dev: {len(dev)} examples")
    print(train[1])


if __name__ == "__main__":
    main()
