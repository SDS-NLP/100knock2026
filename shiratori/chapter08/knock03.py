import pickle

import numpy as np
import torch
import torch.nn as nn
from model import BoWClassifier

embedding_matrix = np.load("data/embedding_matrix.npy")

with open("data/sst2_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("data/sst2_dev_dataset.pkl", "rb") as f:
    dev_dataset = pickle.load(f)


def main():
    model = BoWClassifier(embedding_matrix)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for sample in train_dataset:

            input_ids = sample["input_ids"]
            label = sample["label"]

            optimizer.zero_grad()

            output = model(input_ids)

            loss = criterion(output, label)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")
    torch.save(model.state_dict(), "data/sst2_bow_classifier.pth")


if __name__ == "__main__":
    main()
