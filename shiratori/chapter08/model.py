import torch
import torch.nn as nn


class BoWClassifier(nn.Module):
    def __init__(self, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float),
            freeze=False,
        )

        self.linear = nn.Linear(embedding_matrix.shape[1], 1)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = x.mean(dim=0)

        x = self.linear(x)

        return x
