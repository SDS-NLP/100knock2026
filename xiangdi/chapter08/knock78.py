import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


embedding_matrix_path = "embedding_matrix.pt"
train_dataset_path = "sst_train.pt"
dev_dataset_path = "sst_dev.pt"

batch_size = 32
epochs = 5
lr = 1e-2


class BoWLogisticRegression(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=False,
            padding_idx=0,
            sparse=True
        )

        d_emb = embedding_matrix.size(1)
        self.linear = nn.Linear(d_emb, 1)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        mask = input_ids != 0
        mask = mask.unsqueeze(-1)

        embeddings = embeddings * mask
        bow_vector = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.linear(bow_vector)

        return logits.squeeze(-1)


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


def train(model, train_loader, device, epochs, lr):
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].squeeze(-1).to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"epoch: {epoch + 1}, "
                    f"batch: {batch_idx + 1}, "
                    f"loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        print(f"epoch: {epoch + 1}, average loss: {avg_loss:.4f}")


def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].squeeze(-1).to(device)

            logits = model(input_ids)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

embedding_matrix = torch.load(embedding_matrix_path)
train_dataset = torch.load(train_dataset_path)
dev_dataset = torch.load(dev_dataset_path)

model = BoWLogisticRegression(embedding_matrix)
model = model.to(device)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate
)

train(model, train_loader, device, epochs, lr)

dev_accuracy = evaluate(model, dev_loader, device)
print(f"dev accuracy: {dev_accuracy:.4f}")

torch.save(model.state_dict(), "bow_logistic_regression_finetuned.pt")