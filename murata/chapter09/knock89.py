import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel
from knock86 import collate
from knock87 import SSTDataset

MODEL = "bert-base-uncased"

class MaxPoolClassifier(nn.Module):
    """87の[CLS]分類ヘッドの代わりに、最終層の各トークンの最大値プーリングを使う"""
    def __init__(self, model_name=MODEL, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state
        h = h.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        pooled = h.max(dim=1).values
        return self.fc(self.dropout(pooled))

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for b in loader:
            x = b['input_ids'].to(device)
            m = b['attention_mask'].to(device)
            y = b['label'].to(device)
            logits = model(x, m)
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    train = torch.load('sst_bert_train.pt')
    dev   = torch.load('sst_bert_dev.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_loader = DataLoader(SSTDataset(train), batch_size=32,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(SSTDataset(dev), batch_size=64,
                              shuffle=False, collate_fn=collate)

    model = MaxPoolClassifier().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for ep in range(2):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input_ids'].to(device)
            m = b['attention_mask'].to(device)
            y = b['label'].to(device)
            loss = criterion(model(x, m), y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * y.size(0)
        print(f'ep{ep+1} loss={total_loss/len(train):.4f} '
              f'dev_acc={evaluate(model, dev_loader, device):.4f}')

    torch.save(model.state_dict(), 'bert_maxpool_sst.pt')
    print(f'final dev acc = {evaluate(model, dev_loader, device):.4f}')
