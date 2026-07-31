"""
以下の4文の全ての組み合わせに対して、最終層の埋め込みベクトルの平均を
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


def get_mean_embeddings(sentences, tokenizer, model):
    """
    文のリストをBERTに入力し、各文の最終層埋め込みベクトルの平均を返す

    パディング部分がattention_maskが0の位置を平均計算から除外する。

    戻り値:
        Tensor (num_sentences, hidden_size)
    """
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (num_sentences, seq_len, hidden_size)

    # パディング位置を平均から除外するためのマスクを作成する
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()  # (num_sentences, seq_len, 1)

    summed = (last_hidden_state * attention_mask).sum(dim=1)      # (num_sentences, hidden_size)
    lengths = attention_mask.sum(dim=1).clamp(min=1)              # (num_sentences, 1)
    mean_embeddings = summed / lengths                            # (num_sentences, hidden_size)

    return mean_embeddings


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model.eval()

    mean_embeddings = get_mean_embeddings(TARGET_SENTENCES, tokenizer, model)

    print("--- 対象文 ---")
    for i, sentence in enumerate(TARGET_SENTENCES):
        print(f"{i}: {sentence}")

    print("\n--- 平均埋め込みによるコサイン類似度（全組み合わせ） ---")
    for i, j in combinations(range(len(TARGET_SENTENCES)), 2):
        sim = F.cosine_similarity(mean_embeddings[i], mean_embeddings[j], dim=0).item()
        print(f"({i}, {j}) 類似度={sim:.4f}  | 「{TARGET_SENTENCES[i]}」 - 「{TARGET_SENTENCES[j]}」")


if __name__ == "__main__":
    main()
