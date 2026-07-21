import numpy as np
import torch
import torch.nn as nn

class BoWClassifier(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        weight = torch.tensor(embeddings, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
        self.fc = nn.Linear(weight.shape[1], 1)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        mask = (input_ids != 0).unsqueeze(-1).float()
        summed = (emb * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean = summed / lengths
        return self.fc(mean)

def main():
    embeddings = np.load('embeddings.npy')
    model = BoWClassifier(embeddings)
    print(model)

    dummy = torch.tensor([[3475, 87, 15888, 90], [42637, 0, 0, 0]])
    logits = model(dummy)
    print(f'\nlogits: {logits.detach().squeeze(-1)}')
    print(f'prob:   {torch.sigmoid(logits).detach().squeeze(-1)}')

if __name__ == '__main__':
    main()
