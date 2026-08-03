import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from knock70 import WordEmbeddingToolkit
from knock71 import register_wordex, remap_ids, tokenize
from knock72 import ModelV2
from knock75 import collate

toolkit = WordEmbeddingToolkit()
TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"
df_train = tokenize(TRAIN_PATH, toolkit)
df_dev = tokenize(DEV_PATH, toolkit)
train_input_ids = df_train.get_column("input_ids").to_list()
dev_input_ids = df_dev.get_column("input_ids").to_list()

# データセットで使われている単語を収集
train_wordex = register_wordex(train_input_ids)
dev_wordex = register_wordex(dev_input_ids)
wordex = sorted(list(train_wordex | dev_wordex))
pre_weight = toolkit.matrix[wordex]
id_map = {old: new for new, old in enumerate(wordex)}
X_train = remap_ids(train_input_ids, id_map)
X_dev = remap_ids(dev_input_ids, id_map)

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
input_dim = len(df_train.get_column("mean_vector")[0])
hidden_unit = 5
model_1 = ModelV2(input_dim, hidden_unit, pre_weight, freeze=False).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model_1.parameters(), lr=0.1)
torchmetrics_accuracy = Accuracy(task="binary").to(device)
torch.manual_seed(42)
epochs = 100

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
# Epoch: 0 | Loss: 0.68638, Accuracy: 55.82%
# Dev Loss: 1.16729, Dev Accuracy: 80.96%
