"""89. アーキテクチャの変更

[CLS]トークンの代わりに、各トークンの最終層埋め込みを最大値プーリングして
文ベクトルとする分類モデルを設計し、極性分析タスク向けにファインチューニングする。
検証セット上での正解率を計測する。
"""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from knock85 import DATA_DIR, load_tsv


MODEL_NAME = "bert-base-uncased"
MODEL_PATH = Path(__file__).resolve().parent / "model_knock89.pt"
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
NUM_LABELS = 2


class MaxPoolClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, **inputs: torch.Tensor) -> torch.Tensor:
        last_hidden_state = self.encoder(**inputs).last_hidden_state

        # PAD部分を-infにしてから系列方向に最大値プーリングする
        mask = inputs["attention_mask"].unsqueeze(-1).bool()
        masked_hidden_state = last_hidden_state.masked_fill(~mask, float("-inf"))
        pooled, _ = masked_hidden_state.max(dim=1)

        return self.classifier(pooled)


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


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for encoded, labels in loader:
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(**encoded)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    for encoded, labels in loader:
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = labels.to(device)

        logits = model(**encoded)
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

    model = MaxPoolClassifier(MODEL_NAME, NUM_LABELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        accuracy = evaluate(model, dev_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} dev_accuracy={accuracy:.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    main()
