from pathlib import Path

import torch
from torch import nn

from knock71 import load_embeddings, load_sst2_as_ids
from knock72 import BagOfWordsClassifier


def accuracy(model, examples, device="cpu"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for example in examples:
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            label = example["label"].to(device)
            logit = model(input_ids).view(-1)
            pred = (torch.sigmoid(logit) >= 0.5).float()
            correct += int(pred.item() == label.item())
    return correct / len(examples)


def train_one_by_one(model, train, dev, epochs=1, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(len(train))
        for step, index in enumerate(indices, 1):
            example = train[index]
            input_ids = example["input_ids"].unsqueeze(0).to(device)
            label = example["label"].view(1, 1).to(device)

            optimizer.zero_grad()
            loss = criterion(model(input_ids), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if step % 10000 == 0:
                print(f"epoch={epoch} step={step} loss={total_loss / step:.4f}")

        print(
            f"epoch={epoch} loss={total_loss / len(train):.4f} "
            f"dev_accuracy={accuracy(model, dev, device):.4f}"
        )
    return model


def main():
    embedding_matrix, token_to_id, _ = load_embeddings(limit=100000)
    train, dev = load_sst2_as_ids(token_to_id)
    model = BagOfWordsClassifier(embedding_matrix, freeze=True)
    train_one_by_one(model, train, dev, epochs=1, lr=1e-3)
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "artifacts/knock73_bow.pt")


if __name__ == "__main__":
    main()
