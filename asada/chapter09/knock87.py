from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knock85 import tokenize
from knock86 import collate

TRAIN_PATH = "SST-2/train.tsv"
DEV_PATH = "SST-2/dev.tsv"
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
df_train = tokenize(TRAIN_PATH, tokenizer)
df_dev = tokenize(DEV_PATH, tokenizer)
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
model_1 = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model_1.parameters(), lr=2e-5)
torchmetrics_accuracy = Accuracy(task="binary").to(device)
torch.manual_seed(42)
epochs = 3

model_1.train()
for epoch in range(epochs):
    total_loss = 0
    torchmetrics_accuracy.reset()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).long()
        y_logits = model_1(batch_X).logits
        y_pred = torch.argmax(y_logits, dim=1)
        loss = loss_fn(y_logits, batch_y)
        total_loss += loss.item()
        torchmetrics_accuracy.update(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(
        f"Epoch: {epoch} | Loss: {total_loss / len(train_loader):.5f}, Accuracy: {torchmetrics_accuracy.compute() * 100:.2f}%"
    )

model_1.eval()
with torch.inference_mode():
    total_loss = 0
    torchmetrics_accuracy.reset()
    for batch_X_dev, batch_y_dev in dev_loader:
        batch_X_dev, batch_y_dev = batch_X_dev.to(device), batch_y_dev.to(device).long()
        dev_logits = model_1(batch_X_dev).logits
        dev_pred = torch.argmax(dev_logits, dim=1)
        dev_loss = loss_fn(dev_logits, batch_y_dev)
        total_loss += dev_loss.item()
        torchmetrics_accuracy.update(dev_pred, batch_y_dev)
print(
    f"Dev Loss: {total_loss / len(dev_loader):.5f}, Dev Accuracy: {torchmetrics_accuracy.compute() * 100:.2f}%"
)
# 重みを保存
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_1_PATH = MODEL_PATH / "model_1.pth"
print(f"Saving model to: {MODEL_1_PATH}")
torch.save(obj=model_1.state_dict(), f=MODEL_1_PATH)
# Result
# Epoch: 0 | Loss: 0.40713, Accuracy: 77.04%
# Epoch: 1 | Loss: 0.13398, Accuracy: 95.29%
# Epoch: 2 | Loss: 0.08732, Accuracy: 97.01%
# Dev Loss: 0.22555, Dev Accuracy: 91.74%
# Saving model to: models/model_1.pth
