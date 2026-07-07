import torch
from torch import nn

from knock71 import load_embeddings


class BagOfWordsClassifier(nn.Module):
    def __init__(self, embedding_matrix, freeze=True):
        super().__init__()
        weights = torch.tensor(embedding_matrix, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            weights, freeze=freeze, padding_idx=0
        )
        self.linear = nn.Linear(weights.size(1), 1)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        mask = input_ids.ne(0).unsqueeze(-1)
        vector = (embedded * mask).sum(dim=1)
        length = mask.sum(dim=1).clamp_min(1)
        return self.linear(vector / length)


def main():
    embedding_matrix, _, _ = load_embeddings(limit=100000)
    model = BagOfWordsClassifier(embedding_matrix, freeze=True)
    input_ids = torch.tensor([[3475, 87, 15888, 90, 27695, 42637]])
    logits = model(input_ids)
    probability = torch.sigmoid(logits)
    print(model)
    print(f"logit: {logits.item():.4f}")
    print(f"P(positive): {probability.item():.4f}")


if __name__ == "__main__":
    main()
