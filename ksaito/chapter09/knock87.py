import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_data(path):
    df = pd.read_csv(path, sep="\t")
    return df["sentence"].tolist(), df["label"].tolist()


def create_data_loader(texts, labels, tokenizer, batch_size, shuffle=False):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"],
        torch.tensor(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train(model, data_loader, optimizer, device):
    model.train()
    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss
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
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


def main():
    model_name = "bert-base-uncased"
    batch_size = 32
    epochs = 1
    data_dir = os.path.join(os.path.dirname(__file__), "data", "SST-2")
    model_dir = os.path.join(os.path.dirname(__file__), "models", "knock87")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_texts, train_labels = load_data(os.path.join(data_dir, "train.tsv"))
    dev_texts, dev_labels = load_data(os.path.join(data_dir, "dev.tsv"))
    train_loader = create_data_loader(
        train_texts, train_labels, tokenizer, batch_size, shuffle=True
    )
    dev_loader = create_data_loader(dev_texts, dev_labels, tokenizer, batch_size)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(epochs):
        train(model, train_loader, optimizer, device)
        accuracy = evaluate(model, dev_loader, device)
        print(f"epoch: {epoch + 1}, accuracy: {accuracy:.4f}")

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
