"""72. Bag of wordsモデルの構築

単語埋め込みの平均ベクトルを文表現とし、線形層で
ポジティブ/ネガティブを分類するロジスティック回帰モデルを実装する。
"""

from __future__ import annotations

import torch
from torch import nn

from knock70 import load_embedding_matrix
from knock71 import load_train_dev_datasets


class SentenceLogisticRegression(nn.Module):
    """Average word embeddings and classify with a linear layer."""

    def __init__(self, embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=0,
        )
        self.linear = nn.Linear(embedding_matrix.size(1), 1)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs into mean pooled sentence embeddings."""
        if input_ids.dim() == 1:
            embeddings = self.embedding(input_ids)
            return embeddings.mean(dim=0)

        embeddings = self.embedding(input_ids)
        valid_mask = (input_ids != 0).unsqueeze(-1)
        summed = (embeddings * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1)
        return summed / lengths

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits for binary classification."""
        sentence_vectors = self.encode(input_ids)
        logits = self.linear(sentence_vectors)
        return logits.squeeze(-1)

    def predict_proba(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return probabilities after applying sigmoid."""
        return torch.sigmoid(self(input_ids))


def main() -> None:
    embedding_matrix, token_to_id, _ = load_embedding_matrix()
    train_dataset, _ = load_train_dev_datasets(token_to_id)
    model = SentenceLogisticRegression(embedding_matrix)

    example = train_dataset[0]
    input_ids = example["input_ids"]
    label = example["label"]

    logits = model(input_ids)
    probability = model.predict_proba(input_ids)

    print(model)
    print(f"text: {example['text']}")
    print(f"label: {label.item():.1f}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"logit: {float(logits.item()):.6f}")
    print(f"probability: {float(probability.item()):.6f}")


if __name__ == "__main__":
    main()
