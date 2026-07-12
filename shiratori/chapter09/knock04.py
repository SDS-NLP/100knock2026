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

    mean_vectors = []

    with torch.no_grad():
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")
            outputs = model(**inputs)
            mean_vec = outputs.last_hidden_state.mean(dim=1)
            mean_vectors.append(mean_vec)

    for (i, s1), (j, s2) in combinations(enumerate(sentences), 2):
        sim = F.cosine_similarity(mean_vectors[i], mean_vectors[j]).item()
        print(f'"{s1}"')
        print(f'"{s2}"')
        print(f"Cosine similarity: {sim:.4f}\n")


if __name__ == "__main__":
    main()
