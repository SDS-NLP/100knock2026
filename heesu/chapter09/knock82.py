import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "bert-base-uncased"
SENTENCE = "The movie was full of [MASK]."
TOP_K = 10


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(SENTENCE, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_index = (
        (inputs["input_ids"] == tokenizer.mask_token_id)[0]
        .nonzero(as_tuple=True)[0]
        .item()
    )

    # Softmax over the vocabulary turns logits into a probability distribution.
    probs = torch.softmax(logits[0, mask_index], dim=-1)
    top_probs, top_ids = probs.topk(TOP_K)

    print(f"Sentence: {SENTENCE}")
    print(f"Top-{TOP_K} tokens for [MASK]:")
    for rank, (prob, token_id) in enumerate(zip(top_probs, top_ids), start=1):
        token = tokenizer.decode([token_id])
        print(f"{rank:2d}. {token:<15s} {prob.item():.4f}")


if __name__ == "__main__":
    main()
