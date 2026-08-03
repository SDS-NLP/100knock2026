import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from knock70 import WordEmbeddingToolkit
from knock71 import tokenize
from knock72 import ModelV1
from knock75 import collate

toolkit = WordEmbeddingToolkit()
TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"
df_train = tokenize(TRAIN_PATH, toolkit)
df_dev = tokenize(DEV_PATH, toolkit)
X_train = df_train.get_column("input_ids").to_list()
X_dev = df_dev.get_column("input_ids").to_list()
y_train = df_train.get_column("label").to_list()
y_dev = df_dev.get_column("label").to_list()
train_dataset = list(zip(X_train, y_train))
dev_dataset = list(zip(X_dev, y_dev))
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate)
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = torch.tensor(df_train.get_column("mean_vector").to_list()).shape[1]
model_1 = ModelV1(input_dim, toolkit.matrix).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
torchmetrics_accuracy = Accuracy(task="binary").to(device)
torch.manual_seed(42)
epochs = 1000

model_1.train()
for epoch in range(epochs):
    total_loss = 0
    torchmetrics_accuracy.reset()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        y_logits = model_1(batch_X).squeeze(dim=-1)
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, batch_y)
        total_loss += loss.item()
        torchmetrics_accuracy.update(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} | Loss: {total_loss / len(train_loader):.5f}, Accuracy: {torchmetrics_accuracy.compute() * 100:.2f}%"
        )

model_1.eval()
with torch.inference_mode():
    total_loss = 0
    torchmetrics_accuracy.reset()
    for batch_X_dev, batch_y_dev in dev_loader:
        batch_X_dev, batch_y_dev = batch_X_dev.to(device), batch_y_dev.to(device)
        dev_logits = model_1(batch_X_dev).squeeze(dim=-1)
        dev_pred = torch.round(torch.sigmoid(dev_logits))
        dev_loss = loss_fn(dev_logits, batch_y_dev)
        total_loss += dev_loss.item()
        torchmetrics_accuracy.update(dev_pred, batch_y_dev)
print(
    f"Dev Loss: {total_loss / len(dev_loader):.5f}, Dev Accuracy: {torchmetrics_accuracy.compute() * 100:.2f}%"
)

# Result
# Epoch: 0 | Loss: 0.67677, Accuracy: 56.47%
# Epoch: 100 | Loss: 0.43031, Accuracy: 83.36%
# Epoch: 200 | Loss: 0.40744, Accuracy: 83.88%
# Epoch: 300 | Loss: 0.39801, Accuracy: 84.19%
# Epoch: 400 | Loss: 0.39282, Accuracy: 84.36%
# Epoch: 500 | Loss: 0.39016, Accuracy: 84.44%
# Epoch: 600 | Loss: 0.38854, Accuracy: 84.37%
# Epoch: 700 | Loss: 0.38645, Accuracy: 84.43%
# Epoch: 800 | Loss: 0.38521, Accuracy: 84.52%
# Epoch: 900 | Loss: 0.38446, Accuracy: 84.62%
# Dev Loss: 0.49446, Dev Accuracy: 80.28%
