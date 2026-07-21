import torch
from transformers import AutoModelForMaskedLM

from knock80 import MODEL_NAME, load_tokenizer
from knock81 import mask_probabilities


def main():
    text = "The movie was full of [MASK]."
    tokenizer = load_tokenizer()
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    probabilities = mask_probabilities(text, tokenizer, model)
    top_probabilities, top_token_ids = torch.topk(probabilities, k=10)

    for rank, (token_id, probability) in enumerate(
        zip(top_token_ids.tolist(), top_probabilities.tolist()), start=1
    ):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{rank:2d}: {token:<15} {probability:.6f}")


if __name__ == "__main__":
    main()
