import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of [MASK]."


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_id = logits[0, mask_index].argmax().item()
    print(tokenizer.convert_ids_to_tokens(predicted_id))


if __name__ == "__main__":
    main()