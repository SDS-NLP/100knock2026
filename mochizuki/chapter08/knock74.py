import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

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
        return self.fc(summed / lengths)

def collate(batch):
    input_ids = pad_sequence([ex['input_ids'] for ex in batch],
                             batch_first=True, padding_value=0)
    labels = torch.stack([ex['label'] for ex in batch])
    return input_ids, labels

def accuracy(model, examples):
    loader = DataLoader(examples, batch_size=64, shuffle=False, collate_fn=collate)
    correct, n = 0, 0
    model.eval()
    with torch.no_grad():
        for input_ids, labels in loader:
            pred = (torch.sigmoid(model(input_ids)) > 0.5).float()
            correct += (pred == labels).sum().item()
            n += len(labels)
    return correct / n

def main():
    data = torch.load('dataset.pt')
    embeddings = np.load('embeddings.npy')

    model = BoWClassifier(embeddings)
    model.load_state_dict(torch.load('model73.pt'))

    print(f'train accuracy: {accuracy(model, data["train"]):.4f}')
    print(f'dev   accuracy: {accuracy(model, data["dev"]):.4f}')

if __name__ == '__main__':
    main()
