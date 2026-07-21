"""79. アーキテクチャの変更

平均ベクトル + ロジスティック回帰の代わりに、双方向 LSTM (BiLSTM) で
テキストを符号化する分類器を学習し、開発セットの正解率を求める。

- 問題75の collate はトークン列を長い順に並べるので、
  pack_padded_sequence と相性が良い (パディングを正しく無視できる)。
- 最適化には Adam を用いる。埋め込み (300万語) は固定する: ファインチューニング
  すると Adam の状態だけで埋め込みの 2 倍 (約 7GB) を要し GPU メモリに乗らないため。
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from knock70 import PAD_ID, load_embeddings
from knock71 import DATA_DIR, load_dataset
from knock75 import collate
from knock76 import evaluate_batches

BATCH_SIZE = 64
N_EPOCHS = 10
LR = 1e-3
SEED = 42


class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, freeze=False, padding_idx=PAD_ID,
                 hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        weight = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        dim = weight.size(1)
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(dim, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, input_ids):
        lengths = (input_ids != self.padding_idx).sum(dim=1).cpu()  # (batch,)
        embedded = self.embedding(input_ids)                        # (batch, seq, dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # 最終層の前向き h_n[-2] と後ろ向き h_n[-1] を連結
        h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden*2)
        return self.linear(self.dropout(h_cat))       # (batch, 1)


def train(model, train_data, dev_data, n_epochs=N_EPOCHS, lr=LR,
          batch_size=BATCH_SIZE, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        collate_fn=collate)

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)

        train_loss = running_loss / len(train_data)
        _, dev_acc = evaluate_batches(model, dev_data, device=device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} dev_acc={dev_acc:.4f}")

    return model


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    embedding_matrix, word_to_id, _ = load_embeddings()
    train_data = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    dev_data = load_dataset(f"{DATA_DIR}/dev.tsv", word_to_id)

    model = BiLSTMClassifier(embedding_matrix, freeze=True)
    train(model, train_data, dev_data, device=device)

    _, dev_acc = evaluate_batches(model, dev_data, device=device)
    print(f"\nfinal dev accuracy: {dev_acc:.4f}")
