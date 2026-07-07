import torch

from knock71 import load_embeddings, load_sst2_as_ids
from knock72 import BagOfWordsClassifier
from knock73 import accuracy


def main():
    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    _, dev = load_sst2_as_ids(token_to_id)
    model = BagOfWordsClassifier(embedding_matrix, freeze=True)
    model.load_state_dict(torch.load("artifacts/knock73_bow.pt"))
    print(f"dev_accuracy: {accuracy(model, dev):.4f}")


if __name__ == "__main__":
    main()
