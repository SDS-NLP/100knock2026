"""78. 単語埋め込みのファインチューニング

問題77の学習で、単語埋め込みのパラメータも同時に更新する
(ファインチューニングを行う) ように変更し、開発セットの正解率を求める。

問題72のモデルを freeze=False で構築するだけでよい。
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

    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    # freeze=False で埋め込み行列もファインチューニングの対象にする
    model = BoWClassifier(embedding_matrix, freeze=False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,} (埋め込みも更新)")

    train_minibatch(model, train_data, dev_data, device=device)

    _, dev_acc = evaluate_batches(model, dev_data, device=device)
    print(f"\nfinal dev accuracy: {dev_acc:.4f}")
