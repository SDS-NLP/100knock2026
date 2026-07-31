"""
以下の4文の全ての組み合わせに対して、最終層の[CLS]トークンの埋め込みベクトルを
用いてコサイン類似度を求めるスクリプト

"The movie was full of fun."
"The movie was full of excitement."
"The movie was full of crap."
"The movie was full of rubbish."
"""

from itertools import combinations

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名

TARGET_SENTENCES = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def get_cls_embeddings(sentences, tokenizer, model):
    """
    文のリストをBERTに入力し、各文の最終層[CLS]トークンの埋め込みベクトルを返す

    戻り値:
        Tensor (num_sentences, hidden_size)
    """
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (num_sentences, seq_len, hidden_size)

    # 各系列の先頭トークン（[CLS]）の埋め込みを取り出す
    cls_embeddings = last_hidden_state[:, 0, :]  # (num_sentences, hidden_size)
    return cls_embeddings


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    cls_embeddings = get_cls_embeddings(TARGET_SENTENCES, tokenizer, model)

    print("--- 対象文 ---")
    for i, sentence in enumerate(TARGET_SENTENCES):
        print(f"{i}: {sentence}")

    print("\n--- [CLS]埋め込みによるコサイン類似度（全組み合わせ） ---")
    for i, j in combinations(range(len(TARGET_SENTENCES)), 2):
        sim = F.cosine_similarity(cls_embeddings[i], cls_embeddings[j], dim=0).item()
        print(f"({i}, {j}) 類似度={sim:.4f}  | 「{TARGET_SENTENCES[i]}」 - 「{TARGET_SENTENCES[j]}」")


if __name__ == "__main__":
    main()
