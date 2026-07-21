import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
TEXT = "The movie was full of [MASK]."
TOP_K = 10


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(TEXT, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0].item()

    with torch.no_grad():
        mask_logits = model(**inputs).logits[0, mask_index]

    probabilities = torch.softmax(mask_logits, dim=0)
    top_probabilities, top_ids = torch.topk(probabilities, TOP_K)

    print("rank\ttoken\tprobability")
    for rank, (token_id, probability) in enumerate(
        zip(top_ids.tolist(), top_probabilities.tolist()), start=1
    ):
        token = tokenizer.convert_ids_to_tokens(token_id)
        print(f"{rank}\t{token}\t{probability:.6f}")


if __name__ == "__main__":
    main()