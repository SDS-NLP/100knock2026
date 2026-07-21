import torch
import torch.nn as nn

class BoWClassifier(nn.Module):
    def __init__(self, embedding_matrix, padding_idx=0):
        super().__init__()
        V, d = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True, padding_idx = padding_idx
        )
        self.linear = nn.Linear(d, 1)
    
    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return self.linear(mean)