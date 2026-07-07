import torch
from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    batch = sorted(
        batch,
        key=lambda example: len(example["input_ids"]),
        reverse=True
    )

    input_ids = [example["input_ids"] for example in batch]
    labels = [example["label"] for example in batch]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return {
        "input_ids": input_ids,
        "label": labels
    }

train_dataset = torch.load("sst_train.pt")
batch = collate(train_dataset[:4])
print(batch)