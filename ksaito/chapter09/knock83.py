from itertools import combinations

import torch
import torch.nn.functional as F
from transformers import AutoModel

from knock80 import MODEL_NAME, load_tokenizer


SENTENCES = [
    "The movie was full of fun.",
    "The movie was full of excitement.",
    "The movie was full of crap.",
    "The movie was full of rubbish.",
]


def cls_vectors(sentences, tokenizer, model):
    inputs = tokenizer(sentences, padding=True, return_tensors="pt")
    model.eval()
    with torch.inference_mode():
        hidden_states = model(**inputs).last_hidden_state
    return hidden_states[:, 0, :]


def print_pairwise_cosine_similarities(sentences, vectors):
    normalized = F.normalize(vectors, p=2, dim=1)
    similarities = normalized @ normalized.T
    for i, j in combinations(range(len(sentences)), 2):
        print(f"{i + 1}-{j + 1}: {similarities[i, j].item():.6f}")


def main():
    tokenizer = load_tokenizer()
    model = AutoModel.from_pretrained(MODEL_NAME)
    vectors = cls_vectors(SENTENCES, tokenizer, model)

    for i, sentence in enumerate(SENTENCES, start=1):
        print(f"{i}: {sentence}")
    print_pairwise_cosine_similarities(SENTENCES, vectors)


if __name__ == "__main__":
    main()
