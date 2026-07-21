import torch

from knock71 import load_embeddings, load_sst2_as_ids


def collate(examples):
    examples = sorted(examples, key=lambda x: len(x["input_ids"]), reverse=True)
    max_length = len(examples[0]["input_ids"])
    input_ids = torch.zeros((len(examples), max_length), dtype=torch.long)
    labels = torch.zeros((len(examples), 1), dtype=torch.float32)

    for i, example in enumerate(examples):
        length = len(example["input_ids"])
        input_ids[i, :length] = example["input_ids"]
        labels[i] = example["label"]

    return {"input_ids": input_ids, "label": labels}


def main():
    _, token_to_id, _ = load_embeddings(limit=100000)
    train, _ = load_sst2_as_ids(token_to_id)
    batch = collate(train[:4])
    print(batch)


if __name__ == "__main__":
    main()
