"""82. マスクのtop-k予測

BERTを使い、[MASK]に入る確率が高いトークン上位10個を求める。
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of [MASK]."
TOP_K = 10


def predict_top_k(text: str, k: int = TOP_K) -> list[tuple[str, float]]:
    # [MASK]に入るトークンの上位k件を、その確率とともに返す
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()

    if len(mask_positions) != 1:
        raise ValueError("文には[MASK]を1つだけ含めてください")

    mask_index = int(mask_positions[0, 1])

    with torch.no_grad():
        logits = model(**inputs).logits[0, mask_index]

    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_token_ids = torch.topk(probabilities, k)

    predictions = []
    for token_id, probability in zip(top_token_ids, top_probabilities):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        predictions.append((token, float(probability)))

    return predictions


def main() -> None:
    for rank, (token, probability) in enumerate(predict_top_k(TEXT), start=1):
        print(f"{rank:2d}: {token:<15} {probability:.6f}")


if __name__ == "__main__":
    main()
