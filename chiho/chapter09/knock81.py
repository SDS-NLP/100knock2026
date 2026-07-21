"""81. マスクの予測

BERTを使い、文中の[MASK]に入る確率が最も高いトークンを求める。
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of [MASK]."


def predict_mask(text: str) -> str:
    # [MASK]の位置で予測確率が最大となるトークンを返す
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()

    if len(mask_positions) != 1:
        raise ValueError("文には[MASK]を1つだけ含めてください。")

    # mask_positionsの各行は [バッチ内の位置, 文中の位置] を表す
    mask_index = int(mask_positions[0, 1])

    with torch.no_grad():
        logits = model(**inputs).logits

    # [MASK]位置において、語彙中でlogitが最大のトークンIDを選ぶ
    predicted_token_id = int(logits[0, mask_index].argmax())
    return tokenizer.convert_ids_to_tokens(predicted_token_id)


def main() -> None:
    predicted_token = predict_mask(TEXT)
    print(predicted_token)


if __name__ == "__main__":
    main()
