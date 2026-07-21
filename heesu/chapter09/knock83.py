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


def cls_vector(sentence, tokenizer, model):
    """Return the final-layer hidden state of the [CLS] token (position 0)."""
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # last_hidden_state: (batch, seq_len, hidden); [CLS] is at index 0.
    return outputs.last_hidden_state[0, 0]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    vectors = [cls_vector(s, tokenizer, model) for s in SENTENCES]

    print("Cosine similarity of [CLS] vectors:")
    for i, j in combinations(range(len(SENTENCES)), 2):
        sim = F.cosine_similarity(vectors[i], vectors[j], dim=0).item()
        print(f"  {sim:.4f}  | {SENTENCES[i]!r} <-> {SENTENCES[j]!r}")


if __name__ == "__main__":
    main()
