from pathlib import Path
import argparse
import random

import torch
from torch import nn
from torch import optim

from knock72 import BoWClassifier


def train_model(model, train_dataset, epochs=5, lr=1e-3, seed=42, log_interval=5000):
    random.seed(seed)
    torch.manual_seed(seed)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=lr,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled_dataset = train_dataset.copy()
        random.shuffle(shuffled_dataset)
        total_loss = 0.0

        for step, example in enumerate(shuffled_dataset, start=1):
            optimizer.zero_grad()

            logit = model(example['input_ids']).squeeze(0)
            loss = criterion(logit, example['label'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % log_interval == 0:
                avg_loss = total_loss / step
                print(f'epoch: {epoch}, step: {step}, loss: {avg_loss:.4f}')

        avg_loss = total_loss / len(shuffled_dataset)
        print(f'epoch: {epoch}, average loss: {avg_loss:.4f}')


def main():
    parser = argparse.ArgumentParser(description='BoWモデルを訓練セット上で学習する')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-train-size', type=int, default=None)
    parser.add_argument('--log-interval', type=int, default=5000)
    args = parser.parse_args()

    chapter08_dir = Path(__file__).resolve().parent
    embedding_matrix = torch.load(chapter08_dir / 'embedding_matrix.pt', map_location='cpu')
    train_dataset = torch.load(chapter08_dir / 'train_dataset.pt', map_location='cpu', weights_only=False)

    if args.max_train_size is not None:
        train_dataset = train_dataset[:args.max_train_size]

    model = BoWClassifier(embedding_matrix, freeze_embedding=True)
    print(model)
    print(f'学習データ数: {len(train_dataset)}')
    print(f'単語埋め込みを更新するか: {model.embedding.weight.requires_grad}')

    train_model(
        model,
        train_dataset,
        epochs=args.epochs,
        lr=args.lr,
        log_interval=args.log_interval,
    )

    torch.save(model.state_dict(), chapter08_dir / 'bow_model.pt')
    print(f"モデルを保存しました: {chapter08_dir / 'bow_model.pt'}")


if __name__ == '__main__':
    main()
