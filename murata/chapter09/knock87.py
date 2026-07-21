import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification
from knock86 import collate

MODEL = "bert-base-uncased"

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
            m = b['attention_mask'].to(device)
            y = b['label'].to(device)
            logits = model(input_ids=x, attention_mask=m).logits
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

    model = BertForSequenceClassification.from_pretrained(
        MODEL, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for ep in range(2):
        model.train()
        total_loss = 0
        for b in train_loader:
            x = b['input_ids'].to(device)
            m = b['attention_mask'].to(device)
            y = b['label'].to(device)
            out = model(input_ids=x, attention_mask=m, labels=y)
            opt.zero_grad(); out.loss.backward(); opt.step()
            total_loss += out.loss.item() * y.size(0)
        print(f'ep{ep+1} loss={total_loss/len(train):.4f} '
              f'dev_acc={evaluate(model, dev_loader, device):.4f}')

    model.save_pretrained('bert_sst_finetuned')
    print(f'final dev acc = {evaluate(model, dev_loader, device):.4f}')
