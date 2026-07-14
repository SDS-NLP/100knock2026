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


def mean_vector(sentence, tokenizer, model):
    """Return the attention-masked mean of the final-layer hidden states."""
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]  # (seq_len, hidden)
    mask = inputs["attention_mask"][0].unsqueeze(-1)  # (seq_len, 1)
    # Average only over real (non-padding) token positions.
    summed = (hidden * mask).sum(dim=0)
    count = mask.sum()
    return summed / count


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    vectors = [mean_vector(s, tokenizer, model) for s in SENTENCES]

    print("Cosine similarity of mean-pooled vectors:")
    for i, j in combinations(range(len(SENTENCES)), 2):
        sim = F.cosine_similarity(vectors[i], vectors[j], dim=0).item()
        print(f"  {sim:.4f}  | {SENTENCES[i]!r} <-> {SENTENCES[j]!r}")


if __name__ == "__main__":
    main()
