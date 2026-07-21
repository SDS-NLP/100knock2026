"""
学習済み AvgEmbeddingClassifier を開発セットで評価するスクリプト.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models import KeyedVectors


# 埋め込み行列の構築

def build_embedding_matrix(model_path: str):
    """事前学習済み単語ベクトルから埋め込み行列と語彙辞書を構築する."""
    print("単語埋め込みモデルを読み込み中...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("読み込み完了.")

    V, D = len(model.index_to_key), model.vector_size
    embedding_matrix = np.zeros((V + 1, D), dtype=np.float32)
    embedding_matrix[1:] = model.vectors

    word2id: dict[str, int] = {"<PAD>": 0}
    id2word: dict[int, str] = {0: "<PAD>"}
    for token_id, word in enumerate(model.index_to_key, start=1):
        word2id[word] = token_id
        id2word[token_id] = word

    return embedding_matrix, word2id, id2word


# データ読み込み

def tokenize(text: str, word2id: dict[str, int]) -> list[int]:
    """テキストを語彙内トークンのID列に変換する（語彙外単語はスキップ）."""
    return [word2id[tok] for tok in text.split() if tok in word2id]


def load_sst2(tsv_path: str, word2id: dict[str, int], split_name: str = "") -> list[dict]:
    """
    SST-2 形式の TSV を読み込み, 事例の辞書リストを返す.

    全トークンが語彙外となった事例は除外される.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    dataset, n_removed = [], 0

    for _, row in df.iterrows():
        text      = str(row["sentence"])
        label     = int(row["label"])
        input_ids = tokenize(text, word2id)

        if len(input_ids) == 0:
            n_removed += 1
            continue

        dataset.append({
            "text":      text,
            "label":     torch.tensor([float(label)]),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
        })

    tag = f"[{split_name}] " if split_name else ""
    print(f"{tag}読み込み完了: {len(dataset):,} 件  （除外: {n_removed} 件）")
    return dataset


# モデル定義

class AvgEmbeddingClassifier(nn.Module):
    """単語埋め込みの平均ベクトルを特徴量とするロジスティック回帰モデル."""

    def __init__(self, embedding_matrix: np.ndarray, freeze: bool = True):
        super().__init__()
        _, D = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze,
            padding_idx=0,
        )
        self.fc = nn.Linear(D, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input_ids)  # (seq_len, D)
        h   = emb.mean(dim=0)            # (D,)
        out = torch.sigmoid(self.fc(h))  # (1,)
        return out


# 評価

def evaluate(model: nn.Module, dataset: list[dict]) -> float:
    """
    データセット全体の正解率を返す.

    Parameters
    ----------
    model : nn.Module
    dataset : list[dict]
        load_sst2 が返す事例リスト.

    Returns
    -------
    float
        正解率 (0.0 〜 1.0).
    """
    model.eval()
    n_correct = 0

    with torch.no_grad():
        for example in dataset:
            pred  = model(example["input_ids"])          # (1,)
            pred_label  = 1 if pred.item() >= 0.5 else 0
            true_label  = int(example["label"].item())
            n_correct  += int(pred_label == true_label)

    return n_correct / len(dataset)


# エントリポイント

if __name__ == "__main__":
    model_path      = "GoogleNews-vectors-negative300.bin.gz"
    dev_path        = "SST-2/SST-2/dev.tsv"
    checkpoint_path = "model.pt"

    embedding_matrix, word2id, _ = build_embedding_matrix(model_path)
    dev_dataset = load_sst2(dev_path, word2id, split_name="dev")

    # モデルを初期化し, 学習済み重みを読み込む
    model = AvgEmbeddingClassifier(embedding_matrix, freeze=True)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"チェックポイントを {checkpoint_path} から読み込みました.")

    acc = evaluate(model, dev_dataset)
    print(f"開発セット正解率: {acc:.4f}")