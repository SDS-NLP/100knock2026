import pickle
import numpy as np
import torch
import torch.nn as nn
from model import BoWClassifier

# データの読み込み
embedding_matrix = np.load("data/embedding_matrix.npy")

with open("data/sst2_train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

with open("data/sst2_dev_dataset.pkl", "rb") as f:
    dev_dataset = pickle.load(f)


def main():
    model = BoWClassifier(embedding_matrix)

    sample = train_dataset[0]

    print(sample["text"])
    print(sample["label"])

    output = model(sample["input_ids"])

    print(output)


if __name__ == "__main__":
    main()
