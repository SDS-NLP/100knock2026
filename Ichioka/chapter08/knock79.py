"""
ニューラルネットワークのアーキテクチャを変更し, モデルを学習するスクリプト.

これまでの「単語埋め込みの平均ベクトル」に代えて, 双方向LSTM（BiLSTM）で
文全体を系列としてエンコードし, 最終隠れ状態から極性を分類する.
学習したモデルの開発セットにおける正解率を求める.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader


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


# バッチ化（pack_padded_sequence 用に系列長も返す）

def collate(examples: list[dict]) -> dict:
    """
    複数の事例を長さ降順にソートし, パディングして単一バッチにまとめる.

    pack_padded_sequence（enforce_sorted=True）で使うため, 長さ降順ソートは必須.
    """
    examples = sorted(examples, key=lambda ex: len(ex["input_ids"]), reverse=True)

    input_ids_list = [ex["input_ids"] for ex in examples]
    label_list     = [ex["label"]     for ex in examples]
    lengths        = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)

    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    labels           = torch.stack(label_list)

    return {"input_ids": padded_input_ids, "label": labels, "lengths": lengths}


# モデル定義

class BiLSTMClassifier(nn.Module):
    """
    双方向LSTMで文をエンコードし, 最終隠れ状態から極性を分類するモデル.

    平均埋め込みと異なり語順を考慮できる.
    """

    def __init__(self, embedding_matrix: np.ndarray, hidden_size: int = 128, freeze: bool = True):
        super().__init__()
        _, D = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            freeze=freeze,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(D, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : Tensor (N, L)
            パディング済みトークンID行列（長さ降順ソート済み）.
        lengths : Tensor (N,)
            各系列の実際の長さ（CPU上のLongTensor）.

        Returns
        -------
        Tensor (N, 1)
        """
        emb = self.embedding(input_ids)  # (N, L, D)

        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_directions, N, hidden_size)  最終層の順方向・逆方向隠れ状態
        h_forward  = h_n[0]  # (N, hidden_size)
        h_backward = h_n[1]  # (N, hidden_size)
        h = torch.cat([h_forward, h_backward], dim=1)  # (N, 2*hidden_size)

        out = torch.sigmoid(self.fc(h))  # (N, 1)
        return out


# 学習ループ

def train_minibatch(
    model: nn.Module,
    dataset: list[dict],
    n_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    ミニバッチでモデルを学習する. 各エポック終了時に平均損失と正解率を表示する.
    """
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.BCELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        n_correct  = 0
        n_total    = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            label     = batch["label"].to(device)
            lengths   = batch["lengths"]  # pack_padded_sequence の要求によりCPUのまま渡す

            optimizer.zero_grad()
            pred = model(input_ids, lengths)  # (N, 1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            n_correct  += ((pred >= 0.5).float() == label).sum().item()
            n_total    += input_ids.size(0)

        avg_loss = total_loss / n_total
        accuracy = n_correct  / n_total
        print(f"epoch {epoch:3d} | loss: {avg_loss:.4f} | train acc: {accuracy:.4f}")


# 評価

def evaluate(
    model: nn.Module,
    dataset: list[dict],
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> float:
    """データセット全体の正解率を返す."""
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    n_correct = 0
    n_total   = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            label     = batch["label"].to(device)
            lengths   = batch["lengths"]

            pred = model(input_ids, lengths)
            n_correct += ((pred >= 0.5).float() == label).sum().item()
            n_total   += input_ids.size(0)

    return n_correct / n_total


# エントリポイント

if __name__ == "__main__":
    model_path      = "GoogleNews-vectors-negative300.bin.gz"
    train_path      = "SST-2/SST-2/train.tsv"
    dev_path        = "SST-2/SST-2/dev.tsv"
    checkpoint_path = "model79.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    embedding_matrix, word2id, _ = build_embedding_matrix(model_path)
    train_dataset = load_sst2(train_path, word2id, split_name="train")
    dev_dataset   = load_sst2(dev_path,   word2id, split_name="dev")

    model = BiLSTMClassifier(embedding_matrix, hidden_size=128, freeze=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"学習対象パラメータ数: {trainable:,}\n")

    train_minibatch(model, train_dataset, n_epochs=10, lr=1e-3, batch_size=32, device=device)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"チェックポイントを {checkpoint_path} に保存しました.")

    acc = evaluate(model, dev_dataset, batch_size=32, device=device)
    print(f"開発セット正解率: {acc:.4f}")
