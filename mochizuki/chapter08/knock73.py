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

def main():
    data = torch.load('dataset.pt')
    embeddings = np.load('embeddings.npy')

    model = BoWClassifier(embeddings)
    loader = DataLoader(data['train'], batch_size=64, shuffle=True, collate_fn=collate)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-2)

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for input_ids, labels in loader:
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
            n += len(labels)
        print(f'epoch {epoch:2d}  loss={total_loss / n:.4f}  train_acc={correct / n:.4f}')

    torch.save(model.state_dict(), 'model73.pt')
    print('\nsaved: model73.pt')

if __name__ == '__main__':
    main()
