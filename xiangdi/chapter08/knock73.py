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

def train(model, train_loader, epochs=5, lr=1e-3):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.linear.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
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


embedding_matrix = torch.load("embedding_matrix.pt")
train_dataset = torch.load("sst_train.pt")

model = BoWLogisticRegression(embedding_matrix)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn
)

train(model, train_loader, epochs=5, lr=1e-3)

torch.save(model.state_dict(), "bow_logistic_regression.pt")