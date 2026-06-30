from pathlib import Path

import torch
from torch import nn


class BoWClassifier(nn.Module):
    def __init__(self, embedding_matrix, freeze_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embedding,
            padding_idx=0,
        )
        self.linear = nn.Linear(embedding_matrix.size(1), 1)

    def forward(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        mask = input_ids.ne(0).unsqueeze(-1)
        embeddings = self.embedding(input_ids)
        embeddings = embeddings * mask

        lengths = mask.sum(dim=1).clamp(min=1)
        mean_embeddings = embeddings.sum(dim=1) / lengths
        logits = self.linear(mean_embeddings)

        return logits


def main():
    chapter08_dir = Path(__file__).resolve().parent
    embedding_matrix = torch.load(chapter08_dir / 'embedding_matrix.pt', map_location='cpu')
    train_dataset = torch.load(chapter08_dir / 'train_dataset.pt', map_location='cpu', weights_only=False)

    model = BoWClassifier(embedding_matrix, freeze_embedding=True)
    example = train_dataset[0]

    with torch.no_grad():
        logit = model(example['input_ids'])
        probability = torch.sigmoid(logit)

    print(model)
    print('入力例:')
    print(example)
    print(f'ロジット: {logit.item():.4f}')
    print(f'ポジティブである確率: {probability.item():.4f}')


if __name__ == '__main__':
    main()
