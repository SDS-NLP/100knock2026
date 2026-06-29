import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, embedding_tensor):
        super().__init__() #親クラスの設定の引き継ぎ

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            padding_idx=0,
            freeze=False
        )

        embed_dim = embedding_tensor.size(1)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)          # <PAD>以外を1、<PAD>を0にするマスク
        sum_embeddings = (embedded * mask).sum(dim=1)  # 有効な単語のベクトルだけを足し算
        valid_lengths = mask.sum(dim=1).clamp(min=1)   # 有効な単語数（0割防止のため最小値1）
        
        avg_embedded = sum_embeddings / valid_lengths  # 合計を単語数で割って「平均ベクトル」に
        
        out = self.linear(avg_embedded)
        
        return out

