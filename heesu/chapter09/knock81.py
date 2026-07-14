import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "bert-base-uncased"
SENTENCE = "The movie was full of [MASK]."


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(SENTENCE, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # Locate the [MASK] position and take the arg-max over the vocabulary.
    mask_index = (
        (inputs["input_ids"] == tokenizer.mask_token_id)[0]
        .nonzero(as_tuple=True)[0]
        .item()
    )
    predicted_id = logits[0, mask_index].argmax().item()
    predicted_token = tokenizer.decode([predicted_id])

    print(f"Sentence: {SENTENCE}")
    print(f"Best token for [MASK]: {predicted_token!r}")


if __name__ == "__main__":
    main()
