from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence


BASE_DIR = Path(__file__).resolve().parent


def collate(batch):
    batch = sorted(batch, key=lambda ex: len(ex["input_ids"]), reverse=True)
    input_ids = pad_sequence(
        [ex["input_ids"] for ex in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.stack([ex["label"] for ex in batch])
    return {"input_ids": input_ids, "label": labels}


def main():
    data = torch.load(BASE_DIR / "dataset.pt")
    batch = data["train"][:4]
    print(collate(batch))


if __name__ == "__main__":
    main()
