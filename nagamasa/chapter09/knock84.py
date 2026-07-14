import itertools

import torch.nn.functional as F

from knock80 import load_tokenizer
from knock83 import SENTENCES, encode, load_bert

# 84. 平均による文ベクトル
# 83と同じ4文を、最終層の全トークンの平均で文ベクトル化して比較する。
# [PAD] を平均に混ぜないよう attention_mask で選別する(今回は全文9トークンで
# パディング無しのため素朴平均と一致するが、長さ不揃いのバッチでも壊れない形にする)。


def mean_pool(hidden, attention_mask):
    """実トークンのみの平均 (バッチ, 768)。PADは掛け算で消し、実トークン数で割る。"""
    mask = attention_mask.unsqueeze(-1)  # (B, L) → (B, L, 1) 768次元側へブロードキャスト
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_bert()

    hidden, inputs = encode(SENTENCES, tokenizer, model)
    vecs = mean_pool(hidden, inputs["attention_mask"])

    for i, j in itertools.combinations(range(len(SENTENCES)), 2):
        sim = F.cosine_similarity(vecs[i], vecs[j], dim=0)
        print(f"{SENTENCES[i]:<40} × {SENTENCES[j]:<40} {sim:.4f}")
