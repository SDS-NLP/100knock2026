import itertools

import torch
import torch.nn.functional as F
from transformers import AutoModel

from knock80 import MODEL_NAME, load_tokenizer

# 83. CLSトークンによる文ベクトル
# ヘッド無し本体(AutoModel)の最終層 last_hidden_state (バッチ, 系列長, 768) から
# 先頭([CLS]位置)のベクトルを文ベクトルとし、全ペアのコサイン類似度を出す。
# 4文は padding=True で1バッチにまとめて流す(attentionは事例内で閉じるので結果は1文ずつと同じ)。

SENTENCES = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def load_bert(model_name=MODEL_NAME):
    """ヘッド無しのBERT本体。84でも使う。"""
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return model


def encode(sentences, tokenizer, model):
    """文リストを1バッチで forward。最終層の隠れ状態と入力の辞書を返す。84でも使う。"""
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state, inputs


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = load_bert()

    hidden, _ = encode(SENTENCES, tokenizer, model)
    cls_vecs = hidden[:, 0]  # 各文の位置0 = [CLS] → (4, 768)

    for i, j in itertools.combinations(range(len(SENTENCES)), 2):
        sim = F.cosine_similarity(cls_vecs[i], cls_vecs[j], dim=0)
        print(f"{SENTENCES[i]:<40} × {SENTENCES[j]:<40} {sim:.4f}")
