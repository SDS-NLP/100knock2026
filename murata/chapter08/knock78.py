import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from knock75 import collate
from knock76 import SSTDataset, evaluate

class BoWClassifierFT(nn.Module):
    def __init__(self, E):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(E, freeze=False, padding_idx=0)
        self.fc = nn.Linear(E.size(1), 1)
    def forward(self, input_ids):
        mask = (input_ids != 0).float().unsqueeze(-1)
        h = self.emb(input_ids) * mask
        avg = h.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(avg).squeeze(-1)

if __name__ == '__main__':
    train = torch.load('sst_train.pt')
    dev   = torch.load('sst_dev.pt')
    E     = torch.load('embedding_matrix.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(SSTDataset(train), batch_size=64,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(SSTDataset(dev), batch_size=128,
                              shuffle=False, collate_fn=collate)

    model = BoWClassifierFT(E).to(device)
    # 埋め込みは小さめ、分類器は普通のlr
    opt = torch.optim.Adam([
        {'params': model.emb.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(),  'lr': 1e-3},
    ])

    for ep in range(10):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input_ids'].to(device); y = b['label'].to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0)
        print(f'ep{ep+1} loss={total_loss/len(train):.4f} '
              f'dev_acc={evaluate(model, dev_loader, device):.4f}')

    torch.save(model.state_dict(), 'bow_model_ft.pt')