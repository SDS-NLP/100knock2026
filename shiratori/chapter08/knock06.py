import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from knock05 import collate
from model import BoWClassifier

# 埋め込み行列
embedding_matrix = np.load("data/embedding_matrix.npy")

# データセット
with open("data/sst2_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("data/sst2_dev_dataset.pkl", "rb") as f:
    dev_dataset = pickle.load(f)


def main():

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate)

    dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    # モデル
    model = BoWClassifier(embedding_matrix)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10

    # ===== 学習 =====
    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"]
            label = batch["label"]

            optimizer.zero_grad()

            output = model(input_ids)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # モデル保存
    torch.save(model.state_dict(), "data/sst2_bow_classifier.pth")

    # ===== 評価 =====
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dev_loader:

            input_ids = batch["input_ids"]
            label = batch["label"]

            output = model(input_ids)

            pred = (torch.sigmoid(output) >= 0.5).float()

            correct += (pred == label).sum().item()
            total += label.size(0)

    accuracy = correct / total

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()

# Epoch 1: Loss = 1425.4961
# Epoch 2: Loss = 1413.1560
# Epoch 3: Loss = 1403.6020
# Epoch 4: Loss = 1394.8310
# Epoch 5: Loss = 1386.6619
# Epoch 6: Loss = 1378.5979
# Epoch 7: Loss = 1371.1299
# Epoch 8: Loss = 1363.2958
# Epoch 9: Loss = 1355.2586
# Epoch 10: Loss = 1347.9907
# Accuracy: 0.6078