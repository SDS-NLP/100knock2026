"""71. データセットの読み込み

SST-2 の train/dev を読み込み、Google News の単語ベクトル語彙に基づいて
テキストをトークン ID 列へ変換する。語彙外トークンは無視し、
空列になった事例は除外する。
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from knock70 import load_embedding_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else SCRIPT_DIR / "data" / "SST-2"
TRAIN_PATH = DATA_DIR / "train.tsv"
DEV_PATH = DATA_DIR / "dev.tsv"


def tokenize(text: str) -> list[str]:
    """Tokenize SST-2 text with simple whitespace splitting."""
    return text.split()


def encode_text(text: str, token_to_id: dict[str, int]) -> torch.Tensor:
    """Convert text into a tensor of in-vocabulary token IDs."""
    input_ids = [token_to_id[token] for token in tokenize(text) if token in token_to_id]
    return torch.tensor(input_ids, dtype=torch.long)


def load_dataset(path: Path, token_to_id: dict[str, int]) -> list[dict[str, object]]:
    """Load SST-2 TSV and drop examples that become empty after OOV removal."""
    dataset: list[dict[str, object]] = []

    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            text, label_str = line.rsplit("\t", maxsplit=1)
            input_ids = encode_text(text, token_to_id)
            if input_ids.numel() == 0:
                continue

            dataset.append(
                {
                    "text": text,
                    "label": torch.tensor([float(label_str)], dtype=torch.float32),
                    "input_ids": input_ids,
                }
            )

    return dataset


def load_train_dev_datasets(
    token_to_id: dict[str, int],
    train_path: Path = TRAIN_PATH,
    dev_path: Path = DEV_PATH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Load the train and dev splits."""
    train_dataset = load_dataset(train_path, token_to_id)
    dev_dataset = load_dataset(dev_path, token_to_id)
    return train_dataset, dev_dataset


def summarize_dataset(name: str, dataset: list[dict[str, object]]) -> None:
    """Print basic dataset statistics."""
    lengths = [int(example["input_ids"].numel()) for example in dataset]
    positives = sum(int(float(example["label"][0].item())) for example in dataset)
    negatives = len(dataset) - positives

    print(f"{name} size: {len(dataset)}")
    print(f"{name} positives: {positives}")
    print(f"{name} negatives: {negatives}")
    print(f"{name} min length: {min(lengths)}")
    print(f"{name} max length: {max(lengths)}")
    print(f"{name} first example: {dataset[0]}")


def main() -> None:
    _, token_to_id, _ = load_embedding_matrix()
    train_dataset, dev_dataset = load_train_dev_datasets(token_to_id)

    summarize_dataset("train", train_dataset)
    summarize_dataset("dev", dev_dataset)


if __name__ == "__main__":
    main()
