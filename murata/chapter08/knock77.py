import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from knock72 import BoWClassifier
from knock75 import collate
from knock76 import SSTDataset, evaluate

if __name__ == '__main__':
    train = torch.load('sst_train.pt')
    dev   = torch.load('sst_dev.pt')
    E     = torch.load('embedding_matrix.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    train_loader = DataLoader(SSTDataset(train), batch_size=64,
                              shuffle=True, collate_fn=collate)
    dev_loader   = DataLoader(SSTDataset(dev), batch_size=128,
                              shuffle=False, collate_fn=collate)

    model = BoWClassifier(E).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-1)

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

    torch.save(model.state_dict(), 'bow_model_gpu.pt')