import random, torch, torch.nn as nn
from knock72 import BoWClassifier

train = torch.load('sst_train.pt')
E     = torch.load('embedding_matrix.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BoWClassifier(E).to(device)
criterion = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)

for epoch in range(10):
    model.train()
    random.shuffle(train)
    total = 0.0
    for ex in train:
        x = ex['input_ids'].unsqueeze(0).to(device)
        y = ex['label'].unsqueeze(0).to(device)
        logit = model(x)
        loss = criterion(logit, y)
        optim.zero_grad(); loss.backward(); optim.step()
        total += loss.item()
    print(f'epoch {epoch+1}: loss={total/len(train):.4f}')

torch.save(model.state_dict(), 'bow_model.pt')

"""
epoch 1: loss=0.4390
epoch 2: loss=0.3856
epoch 3: loss=0.3769
epoch 4: loss=0.3731
epoch 5: loss=0.3710
epoch 6: loss=0.3697
epoch 7: loss=0.3690
epoch 8: loss=0.3684
epoch 9: loss=0.3680
epoch 10: loss=0.3676
"""