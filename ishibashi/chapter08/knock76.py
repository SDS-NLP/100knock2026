from pathlib import Path
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from knock72 import BoWClassifier
from knock75 import collate


def train_model(model, train_loader, epochs=5, lr=1e-2):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            logits = model(batch['input_ids'])
            loss = criterion(logits, batch['label'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'epoch: {epoch}, average loss: {avg_loss:.4f}')


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            logits = model(batch['input_ids'])
            preds = (logits >= 0.0).long()
            golds = batch['label'].long()
            correct += (preds == golds).sum().item()
            total += golds.size(0)

    return correct / total, correct, total


def main():
    parser = argparse.ArgumentParser(description='BoWモデルをミニバッチで学習する')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    torch.manual_seed(42)

    chapter08_dir = Path(__file__).resolve().parent
    embedding_matrix = torch.load(chapter08_dir / 'embedding_matrix.pt', map_location='cpu')
    train_dataset = torch.load(chapter08_dir / 'train_dataset.pt', map_location='cpu', weights_only=False)
    dev_dataset = torch.load(chapter08_dir / 'dev_dataset.pt', map_location='cpu', weights_only=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = BoWClassifier(embedding_matrix, freeze_embedding=True)
    print(model)
    print(f'学習データ数: {len(train_dataset)}')
    print(f'バッチサイズ: {args.batch_size}')
    print(f'単語埋め込みを更新するか: {model.embedding.weight.requires_grad}')

    train_model(model, train_loader, epochs=args.epochs, lr=args.lr)

    accuracy, correct, total = evaluate_accuracy(model, dev_loader)
    print(f'開発セットの事例数: {total}')
    print(f'正解数: {correct}')
    print(f'正解率: {accuracy:.4f}')

    torch.save(model.state_dict(), chapter08_dir / 'bow_batch_model.pt')
    print(f"モデルを保存しました: {chapter08_dir / 'bow_batch_model.pt'}")


if __name__ == '__main__':
    main()
