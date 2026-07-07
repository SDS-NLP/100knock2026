from pathlib import Path

import torch

from knock71 import load_embeddings, load_sst2_as_ids
from knock72 import BagOfWordsClassifier
from knock76 import accuracy, train_minibatch


def device_for_gpu():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    device = device_for_gpu()
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    model = BagOfWordsClassifier(embedding_matrix, freeze=True)
    train_minibatch(model, train, dev, device=device, epochs=5, batch_size=256, lr=1e-3)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/knock77_bow_gpu.pt")
    print(f"final_dev_accuracy: {accuracy(model, dev, device=device):.4f}")


if __name__ == "__main__":
    main()
