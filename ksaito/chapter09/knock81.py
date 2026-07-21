import torch
from transformers import AutoModelForMaskedLM

from knock80 import MODEL_NAME, load_tokenizer


def mask_probabilities(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = inputs["input_ids"].eq(tokenizer.mask_token_id).nonzero()
    if mask_positions.size(0) != 1:
        raise ValueError("入力文には[MASK]をちょうど1個含めてください")

    model.eval()
    with torch.inference_mode():
        logits = model(**inputs).logits

    mask_index = mask_positions[0, 1]
    return torch.softmax(logits[0, mask_index], dim=-1)


def main():
    text = "The movie was full of [MASK]."
    tokenizer = load_tokenizer()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    probabilities = mask_probabilities(text, tokenizer, model)
    token_id = probabilities.argmax().item()

    print(tokenizer.convert_ids_to_tokens(token_id))


if __name__ == "__main__":
    main()
