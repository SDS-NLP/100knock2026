import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from knock85 import DATA_DIR, MODEL_NAME, load_split

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 2
LR = 2e-5
MODEL_PATH = "knock87_model"


class SSTDataset(Dataset):
    """Pre-tokenised SST-2 split; padding is deferred to the collate function."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_collate(tokenizer):
    """Build a collate_fn that dynamically pads each minibatch to its longest
    sequence."""

    def collate(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        padded = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = labels
        return padded

    return collate


def build_loader(texts, labels, tokenizer, shuffle):
    dataset = SSTDataset(texts, labels, tokenizer)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=make_collate(tokenizer),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    """Return classification accuracy over a data loader."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / total


def train(model, train_loader, dev_loader, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                avg = running / step
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {avg:.4f}")
        dev_acc = evaluate(model, dev_loader, device)
        print(f"epoch {epoch}: train_loss {running / len(train_loader):.4f}  dev_acc {dev_acc:.4f}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels = load_split(DATA_DIR / "train.tsv")
    dev_texts, dev_labels = load_split(DATA_DIR / "dev.tsv")

    train_loader = build_loader(train_texts, train_labels, tokenizer, shuffle=True)
    dev_loader = build_loader(dev_texts, dev_labels, tokenizer, shuffle=False)

    # Standard sequence-classification head: BERT's [CLS] pooler -> linear layer.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train(model, train_loader, dev_loader, device)

    dev_acc = evaluate(model, dev_loader, device)
    print(f"\nfinal dev accuracy: {dev_acc:.4f}")

    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"saved fine-tuned model to {MODEL_PATH}/")


if __name__ == "__main__":
    main()
