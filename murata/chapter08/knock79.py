# アーキテクチャ変更：TextCNN
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from knock75 import collate
from knock76 import SSTDataset, evaluate

class TextCNN(nn.Module):
    def __init__(self, E, kernel_sizes=(2,3,4), num_filters=100, dropout=0.3):
        super().__init__()
        emb_dim = E.size(1)
        self.emb = nn.Embedding.from_pretrained(E, freeze=False, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, input_ids):
        # (B, L, D) -> (B, D, L)
        h = self.emb(input_ids).transpose(1, 2)
        outs = [F.adaptive_max_pool1d(F.relu(conv(h)), 1).squeeze(-1)
                for conv in self.convs]
        h = torch.cat(outs, dim=1)
        return self.fc(self.dropout(h)).squeeze(-1)

if __name__ == '__main__':
    train = torch.load('sst_train.pt')
    dev   = torch.load('sst_dev.pt')
    E     = torch.load('embedding_matrix.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(SSTDataset(train), batch_size=64,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(SSTDataset(dev), batch_size=128,
                              shuffle=False, collate_fn=collate)

    model = TextCNN(E).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_acc = 0.0
    for ep in range(15):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input_ids'].to(device); y = b['label'].to(device).squeeze(-1)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0)
        acc = evaluate(model, dev_loader, device)
        print(f'ep{ep+1} loss={total_loss/len(train):.4f} dev_acc={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'textcnn_best.pt')
    print(f'best dev acc = {best_acc:.4f}')
    
'''
acc=0.83
'''