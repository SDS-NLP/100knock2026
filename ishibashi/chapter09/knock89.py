"""第9章 knock89: 最大値プーリングを使うBERT極性分類器。"""

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase, get_linear_schedule_with_warmup


MODEL_NAME = "bert-base-uncased"
DATASET_DIR = Path(__file__).with_name("SST-2")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("fine_tuned_bert_sst2_maxpool")


class SST2Dataset(Dataset[tuple[str, int]]):
    def __init__(self, tsv_path: Path) -> None:
        with tsv_path.open(encoding="utf-8", newline="") as file:
            self.examples = [
                (row["sentence"], int(row["label"]))
                for row in csv.DictReader(file, delimiter="\t")
            ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.examples[index]


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def collate_fn(examples: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        texts, labels = zip(*examples)
        batch = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate_fn


class MaxPoolBertClassifier(nn.Module):
    """最終層の非パディングトークンを最大値プーリングして分類する。"""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        hidden_states = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state

        # [PAD] が最大値に選ばれないよう、パディング位置を最小値で埋める。
        padding_mask = ~attention_mask.bool().unsqueeze(-1)
        masked_hidden_states = hidden_states.masked_fill(
            padding_mask, torch.finfo(hidden_states.dtype).min
        )
        sentence_vectors = masked_hidden_states.max(dim=1).values
        logits = self.classifier(self.dropout(sentence_vectors))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return loss, logits


def evaluate(model: MaxPoolBertClassifier, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            _, logits = model(**batch)
            correct += (logits.argmax(dim=1) == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

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
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    train_loader = DataLoader(
        SST2Dataset(DATASET_DIR / "train.tsv"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        SST2Dataset(DATASET_DIR / "dev.tsv"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = MaxPoolBertClassifier(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=args.epochs * len(train_loader)
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            loss, _ = model(**batch)
            assert loss is not None
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if step % args.report_every == 0:
                print(f"epoch={epoch}\tstep={step}/{len(train_loader)}\tloss={total_loss / step:.4f}")

        accuracy = evaluate(model, dev_loader, device)
        print(f"epoch={epoch}\ttrain_loss={total_loss / len(train_loader):.4f}\tdev_accuracy={accuracy:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved model: {args.output_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
