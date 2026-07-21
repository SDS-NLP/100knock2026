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


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(SENTENCES, padding=True, return_tensors="pt")
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    # 各文の先頭位置は、パディングの有無にかかわらず [CLS] である。
    cls_vectors = last_hidden_state[:, 0, :]

    for left, right in combinations(range(len(SENTENCES)), 2):
        similarity = F.cosine_similarity(
            cls_vectors[left].unsqueeze(0), cls_vectors[right].unsqueeze(0)
        ).item()
        print(f"{left + 1}-{right + 1}\t{similarity:.6f}")


if __name__ == "__main__":
    main()