"""
"The movie was full of [MASK]."の"[MASK]"に埋めるのに適切な
トークン上位10個と、その確率（尤度）を求めるスクリプト
"""

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名
TARGET_SENTENCE = "The movie was full of [MASK]."
TOP_K = 10


def main():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(TARGET_SENTENCE, return_tensors="pt")

    # 入力系列中の[MASK]トークンの位置を特定する
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # [MASK]位置のロジットをsoftmaxで確率に変換し、上位TOP_K件を取得する
    mask_logits = logits[0, mask_index, :].squeeze(0)  # (vocab_size,)
    mask_probs = F.softmax(mask_logits, dim=-1)
    top_probs, top_ids = torch.topk(mask_probs, TOP_K)

    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())

    print(f"文: {TARGET_SENTENCE}\n")
    print(f"[MASK]に適切なトークン上位{TOP_K}件:")
    for rank, (token, prob) in enumerate(zip(top_tokens, top_probs.tolist()), start=1):
        print(f"{rank:2d}. {token:<15s} 確率={prob:.4f}")


if __name__ == "__main__":
    main()
