from pathlib import Path

import torch

from knock72 import BoWClassifier


def evaluate_accuracy(model, dataset):
    model.eval()
    correct = 0

    with torch.no_grad():
        for example in dataset:
            logit = model(example['input_ids'])
            pred = int(logit.item() >= 0.0)
            gold = int(example['label'].item())
            correct += int(pred == gold)

    return correct / len(dataset), correct, len(dataset)


def main():
    chapter08_dir = Path(__file__).resolve().parent
    embedding_matrix = torch.load(chapter08_dir / 'embedding_matrix.pt', map_location='cpu')
    dev_dataset = torch.load(chapter08_dir / 'dev_dataset.pt', map_location='cpu', weights_only=False)

    model = BoWClassifier(embedding_matrix, freeze_embedding=True)
    state_dict = torch.load(chapter08_dir / 'bow_model.pt', map_location='cpu')
    model.load_state_dict(state_dict)

    accuracy, correct, total = evaluate_accuracy(model, dev_dataset)

    print(f'開発セットの事例数: {total}')
    print(f'正解数: {correct}')
    print(f'正解率: {accuracy:.4f}')


if __name__ == '__main__':
    main()
