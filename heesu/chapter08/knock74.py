"""74. モデルの評価

問題73で学習したモデルの、開発セット (および訓練セット) における正解率を求める。
問題73で保存した knock73_model.pt を読み込んで評価する。
"""

import torch

from knock70 import load_embeddings
from knock71 import DATA_DIR, load_dataset
from knock72 import BoWClassifier
from knock73 import MODEL_PATH, evaluate

if __name__ == "__main__":
    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    model = BoWClassifier(embedding_matrix, freeze=True)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"loaded model from {MODEL_PATH}")

    train_loss, train_acc = evaluate(model, train_data)
    dev_loss, dev_acc = evaluate(model, dev_data)

    print(f"train: loss={train_loss:.4f} accuracy={train_acc:.4f}")
    print(f"dev:   loss={dev_loss:.4f} accuracy={dev_acc:.4f}")
