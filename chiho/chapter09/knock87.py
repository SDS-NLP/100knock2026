"""87. ファインチューニング

訓練セットを用いて事前学習済みBERTを極性分析タスク向けにファインチューニングし、
検証セット上での正解率を計測する。
"""

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knock85 import DATA_DIR, load_tsv


MODEL_NAME = "bert-base-uncased"
MODEL_DIR = Path(__file__).resolve().parent / "model_knock87"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128


class SST2Dataset(Dataset):
    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.examples[index]


def collate_fn(
    batch: list[tuple[str, int]], tokenizer
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return encoded, torch.tensor(labels)


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for encoded, labels in loader:
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(**encoded, labels=labels)
        outputs.loss.backward()
        optimizer.step()

        total_loss += outputs.loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    for encoded, labels in loader:
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)

        logits = model(**encoded).logits
        predictions = logits.argmax(dim=-1)
        correct += (predictions == labels).sum().item()
    return correct / len(loader.dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = load_tsv(DATA_DIR / "train.tsv")
    dev_examples = load_tsv(DATA_DIR / "dev.tsv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(
        SST2Dataset(train_examples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        SST2Dataset(dev_examples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        accuracy = evaluate(model, dev_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} dev_accuracy={accuracy:.4f}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
