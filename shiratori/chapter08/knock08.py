import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


embedding_matrix = np.load("/content/embedding_matrix.npy")

# data
with open("/content/sst2_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("/content/sst2_dev_dataset.pkl", "rb") as f:
    dev_dataset = pickle.load(f)


class BoWClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=False,
        )

        self.linear = nn.Linear(embedding_matrix.shape[1], 1)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        # BoW→平均
        x = x.mean(dim=1)

        x = self.linear(x)

        return x


def collate(batch):

    batch = sorted(batch, key=lambda x: len(x["input_ids"]), reverse=True)

    max_len = len(batch[0]["input_ids"])

    input_ids = []

    for sample in batch:

        ids = sample["input_ids"]

        pad_len = max_len - len(ids)

        padded = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])

        input_ids.append(padded)

    input_ids = torch.stack(input_ids)

    labels = torch.stack([sample["label"] for sample in batch])

    return {
        "input_ids": input_ids,
        "label": labels,
    }


def main():

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate,
    )

    model = BoWClassifier(embedding_matrix).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10

    # train
    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            output = model(input_ids)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

    # save
    torch.save(model.state_dict(), "/content/sst2_bow_classifier_gpu.pth")

    # eval
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dev_loader:

            input_ids = batch["input_ids"].to(device)
            label = batch["label"].to(device)

            output = model(input_ids)

            pred = (torch.sigmoid(output) >= 0.5).float()

            correct += (pred == label).sum().item()
            total += label.size(0)

    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

# Device: cuda
# Tesla T4
# Epoch 1: Loss = 1424.5051
# Epoch 2: Loss = 1413.1947
# Epoch 3: Loss = 1403.4832
# Epoch 4: Loss = 1395.0820
# Epoch 5: Loss = 1386.5903
# Epoch 6: Loss = 1378.9735
# Epoch 7: Loss = 1371.0820
# Epoch 8: Loss = 1363.4653
# Epoch 9: Loss = 1355.6928
# Epoch 10: Loss = 1347.8009
# Accuracy: 0.6330
