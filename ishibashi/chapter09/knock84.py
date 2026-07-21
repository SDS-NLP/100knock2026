from itertools import combinations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "bert-base-uncased"
SENTENCES = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """パディングを除外して、各文のトークン埋め込みを平均する。"""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(SENTENCES, padding=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    sentence_vectors = mean_pooling(last_hidden_state, inputs["attention_mask"])

    for left, right in combinations(range(len(SENTENCES)), 2):
        similarity = F.cosine_similarity(
            sentence_vectors[left].unsqueeze(0), sentence_vectors[right].unsqueeze(0)
        ).item()
        print(f"{left + 1}-{right + 1}\t{similarity:.6f}")


if __name__ == "__main__":
    main()