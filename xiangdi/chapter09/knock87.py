import csv
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


base_dir = Path(__file__).resolve().parent
train_path = "/Users/caitlyn/Downloads/SST-2/train.tsv"
dev_path = "/Users/caitlyn/Downloads/SST-2/dev.tsv"
model_name = "bert-base-uncased"
output_dir = base_dir / "bert_sst2_model"

epochs = 1
batch_size = 16
lr = 2e-5
max_length = 128


class SSTDataset(Dataset):
    def __init__(self, path):
        self.examples = []

        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                self.examples.append({
                    "text": row["sentence"],
                    "label": int(row["label"]),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(batch, tokenizer, max_length):
    texts = [example["text"] for example in batch]
    labels = [example["label"] for example in batch]

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)

    return inputs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(data_loader, start=1):
        batch = {key: value.to(device) for key, value in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"epoch: {epoch}, step: {step}, loss: {loss.item():.4f}")

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]

            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


torch.manual_seed(0)

device = get_device()
print("device:", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = SSTDataset(train_path)
dev_dataset = SSTDataset(dev_path)

print("train examples:", len(train_dataset))
print("dev examples:", len(dev_dataset))

collate_fn = partial(
    collate,
    tokenizer=tokenizer,
    max_length=max_length,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, device, epoch)
    dev_accuracy = evaluate(model, dev_loader, device)

    print(f"epoch: {epoch}, train loss: {train_loss:.4f}")
    print(f"dev accuracy: {dev_accuracy:.4f}")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("saved model:", output_dir)

# dev accuracy: 0.9278