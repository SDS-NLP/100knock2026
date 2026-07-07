import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BoWLogisticRegression(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0
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


def collate_fn(batch):
    input_ids = [example["input_ids"] for example in batch]
    labels = [example["label"] for example in batch]

    input_ids = nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels).squeeze(-1)

    return input_ids, labels


def evaluate(model, data_loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in data_loader:
            logits = model(input_ids)
            probs = torch.sigmoid(logits)

            preds = (probs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    return accuracy


embedding_matrix = torch.load("embedding_matrix.pt")
dev_dataset = torch.load("sst_dev.pt")

model = BoWLogisticRegression(embedding_matrix)
model.load_state_dict(torch.load("bow_logistic_regression.pt"))

dev_loader = DataLoader(
    dev_dataset,
    batch_size=64,
    shuffle=False,
    collate_fn=collate_fn
)

accuracy = evaluate(model, dev_loader)

print(f"dev accuracy: {accuracy:.4f}")