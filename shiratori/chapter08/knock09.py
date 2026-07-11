import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ==========================
# GPU設定
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


# ==========================
# 埋め込み行列
# ==========================
embedding_matrix = np.load("/content/embedding_matrix.npy")

# ==========================
# データセット
# ==========================
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

        dim = embedding_matrix.shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        # BoWなので平均を取る
        x = x.mean(dim=1)

        x = self.classifier(x)

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

    # ==========================
    # DataLoader
    # ==========================
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

    # ==========================
    # モデル
    # ==========================
    model = BoWClassifier(embedding_matrix).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10

    # ==========================
    # 学習
    # ==========================
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

        # ===== Train Accuracy =====
        model.eval()

        train_correct = 0
        train_total = 0

        with torch.no_grad():
            for batch in train_loader:

                input_ids = batch["input_ids"].to(device)
                label = batch["label"].to(device)

                output = model(input_ids)

                pred = (torch.sigmoid(output) >= 0.5).float()

                train_correct += (pred == label).sum().item()
                train_total += label.size(0)

        train_acc = train_correct / train_total

        # ===== Dev Accuracy =====
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for batch in dev_loader:

                input_ids = batch["input_ids"].to(device)
                label = batch["label"].to(device)

                output = model(input_ids)

                pred = (torch.sigmoid(output) >= 0.5).float()

                dev_correct += (pred == label).sum().item()
                dev_total += label.size(0)

        dev_acc = dev_correct / dev_total

        print(f"Epoch {epoch+1}: " f"Loss={total_loss:.4f}, " f"Train Acc={train_acc:.4f}, " f"Dev Acc={dev_acc:.4f}")


if __name__ == "__main__":
    main()

# Device: cuda
# Tesla T4
# Epoch 1: Loss=1431.2797, Train Acc=0.5582, Dev Acc=0.5092
# Epoch 2: Loss=1428.7579, Train Acc=0.5582, Dev Acc=0.5092
# Epoch 3: Loss=1427.8625, Train Acc=0.5582, Dev Acc=0.5092
# Epoch 4: Loss=1426.3369, Train Acc=0.5582, Dev Acc=0.5092
# Epoch 5: Loss=1423.5775, Train Acc=0.5582, Dev Acc=0.5092
# Epoch 6: Loss=1418.3559, Train Acc=0.5584, Dev Acc=0.5092
# Epoch 7: Loss=1407.8055, Train Acc=0.5791, Dev Acc=0.5424
# Epoch 8: Loss=1384.3614, Train Acc=0.6082, Dev Acc=0.6009
# Epoch 9: Loss=1325.0073, Train Acc=0.7068, Dev Acc=0.7213
# Epoch 10: Loss=1195.5752, Train Acc=0.6480, Dev Acc=0.6548
