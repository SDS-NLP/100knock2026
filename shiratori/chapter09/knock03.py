from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from itertools import combinations


def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    sentences = [
        "The movie was full of fun.",
        "The movie was full of excitement.",
        "The movie was full of crap.",
        "The movie was full of rubbish.",
    ]

    cls_vectors = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            cls_vectors.append(cls)

    for (i, s1), (j, s2) in combinations(enumerate(sentences), 2):
        sim = F.cosine_similarity(cls_vectors[i], cls_vectors[j]).item()
        print(f'"{s1}"')
        print(f'"{s2}"')
        print(f"Cosine similarity: {sim:.4f}\n")


if __name__ == "__main__":
    main()

# "The movie was full of fun."
# "The movie was full of excitement."
# Cosine similarity: 0.9881

# "The movie was full of fun."
# "The movie was full of crap."
# Cosine similarity: 0.9558

# "The movie was full of fun."
# "The movie was full of rubbish."
# Cosine similarity: 0.9475

# "The movie was full of excitement."
# "The movie was full of crap."
# Cosine similarity: 0.9541

# "The movie was full of excitement."
# "The movie was full of rubbish."
# Cosine similarity: 0.9487

# "The movie was full of crap."
# "The movie was full of rubbish."
# Cosine similarity: 0.9807
