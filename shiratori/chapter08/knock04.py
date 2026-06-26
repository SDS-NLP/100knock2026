import pickle

import numpy as np
import torch
import torch.nn as nn
from model import BoWClassifier

embedding_matrix = np.load("data/embedding_matrix.npy")

with open("data/sst2_dev_dataset.pkl", "rb") as f:
    dev_dataset = pickle.load(f)


def main():
    model = BoWClassifier(embedding_matrix)

    model.load_state_dict(torch.load("data/sst2_bow_classifier.pth"))

    model.eval()

    correct = 0

    with torch.no_grad():

        for sample in dev_dataset:

            input_ids = sample["input_ids"]
            label = sample["label"]

            output = model(input_ids)

            pred = torch.sigmoid(output)

            pred_label = 1 if pred >= 0.5 else 0

            if pred_label == int(label.item()):
                correct += 1

    accuracy = correct / len(dev_dataset)

    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
