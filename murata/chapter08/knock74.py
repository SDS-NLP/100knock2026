import torch
from knock72 import BoWClassifier

dev = torch.load('sst_dev.pt')
E   = torch.load('embedding_matrix.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BoWClassifier(E).to(device)
model.load_state_dict(torch.load('bow_model.pt'))
model.eval()

correct = 0
with torch.no_grad():
    for ex in dev:
        x = ex['input_ids'].unsqueeze(0).to(device)
        y = ex['label'].unsqueeze(0).to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).float()
        correct += (pred == y).sum().item()
print(f'dev acc = {correct/len(dev):.4f}')


"""
dev acc = 0.8028
"""

