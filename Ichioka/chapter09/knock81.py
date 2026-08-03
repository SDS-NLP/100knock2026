"""
"The movie was full of [MASK]."の"[MASK]"を埋めるのに
最も適切なトークンを求めるスクリプト
"""

import torch
from transformers import BertTokenizer, BertForMaskedLM


# ===== ハイパーパラメータ =====
MODEL_NAME = "bert-base-uncased"   # 事前学習済みモデル名
TARGET_SENTENCE = "The movie was full of [MASK]."


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

    # [MASK]位置のロジットから最もスコアの高いトークンIDを取得する
    mask_logits = logits[0, mask_index, :]  # (1, vocab_size)
    predicted_id = torch.argmax(mask_logits, dim=-1)
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_id.tolist())[0]

    print(f"文       : {TARGET_SENTENCE}")
    print(f"予測トークン: {predicted_token}")
    print(f"補完後の文  : {TARGET_SENTENCE.replace('[MASK]', predicted_token)}")


if __name__ == "__main__":
    main()
