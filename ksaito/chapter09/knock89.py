import os

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from knock87 import create_data_loader, load_data


class SentimentModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 2),
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = hidden.masked_fill(attention_mask.unsqueeze(-1) == 0, -torch.inf)
        pooled = hidden.max(dim=1).values
        return self.classifier(pooled)


def train(model, data_loader, optimizer, criterion, device):
    model.train()
    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            predictions = model(input_ids, attention_mask).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    model_name = "bert-base-uncased"
    batch_size = 32
    epochs = 1
    data_dir = os.path.join(os.path.dirname(__file__), "data", "SST-2")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_texts, train_labels = load_data(os.path.join(data_dir, "train.tsv"))
    dev_texts, dev_labels = load_data(os.path.join(data_dir, "dev.tsv"))
    train_loader = create_data_loader(
        train_texts, train_labels, tokenizer, batch_size, shuffle=True
    )
    dev_loader = create_data_loader(dev_texts, dev_labels, tokenizer, batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentimentModel(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        accuracy = evaluate(model, dev_loader, device)
        print(f"epoch: {epoch + 1}, accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
