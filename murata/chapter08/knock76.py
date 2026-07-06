import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from knock72 import BoWClassifier
from knock75 import collate

class SSTDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            x = b['input_ids'].to(device)
            y = b['label'].to(device).squeeze(-1)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    train = torch.load('sst_train.pt')
    dev   = torch.load('sst_dev.pt')
    E     = torch.load('embedding_matrix.pt')
    device = torch.device('cpu')

    train_loader = DataLoader(SSTDataset(train), batch_size=32,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(SSTDataset(dev), batch_size=64,
                              shuffle=False, collate_fn=collate)

    model = BoWClassifier(E).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)

    for ep in range(10):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input_ids'].to(device)
            y = b['label'].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0)
        acc = evaluate(model, dev_loader, device)
        print(f'ep{ep+1} loss={total_loss/len(train):.4f} dev_acc={acc:.4f}')

    torch.save(model.state_dict(), 'bow_model_mb.pt')
    print(f'final dev acc = {evaluate(model, dev_loader, device):.4f}')