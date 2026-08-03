import torch
from torch import nn
from torchmetrics import Accuracy

from knock70 import WordEmbeddingToolkit
from knock71 import tokenize
from knock72 import ModelV0

toolkit = WordEmbeddingToolkit()
TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"
df_train = tokenize(TRAIN_PATH, toolkit)
df_dev = tokenize(DEV_PATH, toolkit)

X_train = torch.tensor(df_train.get_column("mean_vector").to_list())
X_dev = torch.tensor(df_dev.get_column("mean_vector").to_list())
y_train = torch.tensor(df_train.get_column("label").to_list(), dtype=torch.float32)
y_dev = torch.tensor(df_dev.get_column("label").to_list(), dtype=torch.float32)

print(f"Shape of X:{X_train.shape}, Shape of y:{y_train.shape}")
# Shape of X:torch.Size([66650, 300]), Shape of y:torch.Size([66650])
device = "cuda" if torch.cuda.is_available() else "cpu"
input_dim = X_train.shape[1]
model_0 = ModelV0(input_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_0.parameters(), lr=0.1)
torchmetrics_accuracy = Accuracy(task="binary").to(device)
torch.manual_seed(42)
epochs = 1000
X_train, y_train = X_train.to(device), y_train.to(device)
X_dev, y_dev = X_dev.to(device), y_dev.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train)
    acc = torchmetrics_accuracy(y_pred, y_train) * 100
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        model_0.eval()
        with torch.inference_mode():
            dev_logits = model_0(X_dev).squeeze()
            dev_pred = torch.round(torch.sigmoid(dev_logits))
            dev_loss = loss_fn(dev_logits, y_dev)
            dev_acc = torchmetrics_accuracy(dev_pred, y_dev) * 100
        print(
            f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Dev Loss: {dev_loss:.5f}, Dev Accuracy: {dev_acc:.2f}%"
        )

model_0.eval()
with torch.inference_mode():
    dev_logits = model_0(X_dev).squeeze()
    dev_pred = torch.round(torch.sigmoid(dev_logits))
    dev_loss = loss_fn(dev_logits, y_dev)
    dev_acc = torchmetrics_accuracy(dev_pred, y_dev) * 100
print(f"Dev Loss: {dev_loss:.5f}, Dev Accuracy: {dev_acc:.2f}%")

# Result
# Dev Loss: 0.56609, Dev Accuracy: 74.89%
