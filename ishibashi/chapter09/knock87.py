import argparse
import csv
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)


MODEL_NAME = "bert-base-uncased"
DATASET_DIR = Path(__file__).with_name("SST-2")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("fine_tuned_bert_sst2")


class SST2Dataset(Dataset[tuple[str, int]]):
    def __init__(self, tsv_path: Path) -> None:
        self.examples: list[tuple[str, int]] = []
        with tsv_path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file, delimiter="\t"):
                self.examples.append((row["sentence"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.examples[index]


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def collate_fn(examples: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*examples)
        batch = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate_fn


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--log-interval", type=int, default=50, help="進捗を表示するミニバッチ間隔")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.log_interval < 1:
        parser.error("--log-interval must be at least 1")

    torch.manual_seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_dataset = SST2Dataset(DATASET_DIR / "train.tsv")
    dev_dataset = SST2Dataset(DATASET_DIR / "dev.tsv")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(
        f"train_examples={len(train_dataset)}\tdev_examples={len(dev_dataset)}\t"
        f"batches_per_epoch={len(train_loader)}",
        flush=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(train_loader),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if step % args.log_interval == 0 or step == len(train_loader):
                elapsed_seconds = time.perf_counter() - epoch_start
                steps_per_second = step / elapsed_seconds
                remaining_seconds = (len(train_loader) - step) / steps_per_second
                print(
                    f"epoch={epoch}/{args.epochs}\t"
                    f"step={step}/{len(train_loader)} ({step / len(train_loader):.1%})\t"
                    f"loss={total_loss / step:.4f}\t"
                    f"elapsed={elapsed_seconds / 60:.1f}min\t"
                    f"eta={remaining_seconds / 60:.1f}min",
                    flush=True,
                )

        accuracy = evaluate(model, dev_loader, device)
        print(
            f"epoch={epoch}\ttrain_loss={total_loss / len(train_loader):.4f}\t"
            f"dev_accuracy={accuracy:.4f}\tepoch_time={(time.perf_counter() - epoch_start) / 60:.1f}min",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model: {args.output_dir}")


if __name__ == "__main__":
    main()