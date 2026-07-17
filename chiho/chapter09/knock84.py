"""84. 平均による文ベクトル

最終層のトークン埋め込みの平均を文ベクトルとし、類似度を求める。
"""

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


def get_mean_vectors(sentences: list[str]) -> torch.Tensor:
    # PADを除く最終層の埋め込みを平均し、各文のベクトルを返す
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    inputs = tokenizer(sentences, padding=True, return_tensors="pt")

    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    # attention_maskは通常トークンで1、PADで0になる。
    mask = inputs["attention_mask"].unsqueeze(-1)
    summed_vectors = (last_hidden_state * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1)
    return summed_vectors / token_counts


def cosine_similarities(
    sentences: list[str], vectors: torch.Tensor
) -> list[tuple[str, str, float]]:
    # 全ての異なる2文の組についてコサイン類似度を返す
    results = []
    for i, j in combinations(range(len(sentences)), 2):
        similarity = F.cosine_similarity(
            vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)
        ).item()
        results.append((sentences[i], sentences[j], similarity))
    return results


def main() -> None:
    vectors = get_mean_vectors(SENTENCES)
    for sentence1, sentence2, similarity in cosine_similarities(
        SENTENCES, vectors
    ):
        print(f"{similarity:.6f}\t{sentence1} / {sentence2}")


if __name__ == "__main__":
    main()
