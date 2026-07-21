from pathlib import Path

import torch

from knock71 import load_embeddings, load_sst2_as_ids
from knock72 import BagOfWordsClassifier
from knock76 import accuracy, train_minibatch
from knock77 import device_for_gpu


def main():
    device = device_for_gpu()
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    model = BagOfWordsClassifier(embedding_matrix, freeze=False)
    train_minibatch(model, train, dev, device=device, epochs=5, batch_size=256, lr=5e-4)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/knock78_bow_finetune.pt")
    print(f"final_dev_accuracy: {accuracy(model, dev, device=device):.4f}")


if __name__ == "__main__":
    main()
