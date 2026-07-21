"""77. GPU 上での学習

問題76のミニバッチ学習を GPU 上で実行し、開発セットの正解率を求める。
train_minibatch / evaluate_batches に device="cuda" を渡すだけでよい。
"""

import random

import torch

from knock70 import load_embeddings
from knock71 import DATA_DIR, load_dataset
from knock72 import BoWClassifier
from knock76 import evaluate_batches, train_minibatch

SEED = 42

if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    model = BoWClassifier(embedding_matrix, freeze=True)
    train_minibatch(model, train_data, dev_data, device=device)

    _, dev_acc = evaluate_batches(model, dev_data, device=device)
    print(f"\nfinal dev accuracy: {dev_acc:.4f}")
