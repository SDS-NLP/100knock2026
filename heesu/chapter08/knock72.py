"""72. Bag of words モデルの構築

単語埋め込みの平均ベクトルでテキストの特徴ベクトルを表現し、
重みベクトルとの内積でポジティブ/ネガティブを分類する
ニューラルネットワーク (ロジスティック回帰モデル) を設計する。
"""

import torch
import torch.nn as nn

from knock70 import PAD_ID


class BoWClassifier(nn.Module):
    """単語埋め込みの平均ベクトル -> 線形層 (1出力) のロジスティック回帰。

    forward は logit を返す (sigmoid は損失関数 BCEWithLogitsLoss 側で適用)。
    """

    def __init__(self, embedding_matrix, freeze=True, padding_idx=PAD_ID):
        super().__init__()
        weight = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        self.padding_idx = padding_idx
        # freeze=True で埋め込み行列を固定 (ファインチューニングしない)
        self.embedding = nn.Embedding.from_pretrained(
            weight, freeze=freeze, padding_idx=padding_idx
        )
        self.linear = nn.Linear(weight.size(1), 1)

    def forward(self, input_ids):
        """input_ids: LongTensor, shape (batch, seq_len)。

        PAD (=0) を除いた実トークンの平均埋め込みベクトルを計算し、
        線形層を通して logit (batch, 1) を返す。
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # (seq,) -> (1, seq)

        mask = (input_ids != self.padding_idx).float()  # (batch, seq)
        embedded = self.embedding(input_ids)            # (batch, seq, dim)
        summed = (embedded * mask.unsqueeze(-1)).sum(dim=1)  # (batch, dim)
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1)
        mean = summed / lengths                          # (batch, dim) = 平均ベクトル
        return self.linear(mean)                         # (batch, 1) = logit


if __name__ == "__main__":
    from knock70 import load_embeddings
    from knock71 import DATA_DIR, load_dataset

    embedding_matrix, word_to_id, _ = load_embeddings()
    model = BoWClassifier(embedding_matrix, freeze=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("model:", model)
    print(f"total params:     {n_params:,}")
    print(f"trainable params: {n_trainable:,} (埋め込みは固定)")

    # 1 事例を通して出力 (logit と確率) を確認
    train = load_dataset(f"{DATA_DIR}/train.tsv", word_to_id)
    ex = train[0]
    with torch.no_grad():
        logit = model(ex["input_ids"])
        prob = torch.sigmoid(logit)
    print("\nexample text:", ex["text"])
    print("logit:", logit.item(), " prob(positive):", prob.item())
