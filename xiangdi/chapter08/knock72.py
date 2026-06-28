import torch
import torch.nn as nn


class BoWLogisticRegression(nn.Module):
    def __init__(self, embedding_matrix, freeze_embeddings=True):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=0
        )

        d_emb = embedding_matrix.size(1)
        self.linear = nn.Linear(d_emb, 1)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)

        if input_ids.dim() == 1:
            bow_vector = embeddings.mean(dim=0)
        else:
            mask = input_ids != 0
            mask = mask.unsqueeze(-1)

            embeddings = embeddings * mask
            bow_vector = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        logits = self.linear(bow_vector)

        return logits.squeeze(-1)

def build_model(embedding_matrix_path):
    embedding_matrix = torch.load(embedding_matrix_path)
    model = BoWLogisticRegression(embedding_matrix)
    return model