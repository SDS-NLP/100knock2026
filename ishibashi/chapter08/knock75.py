from pathlib import Path

import torch


def collate(batch):
    sorted_batch = sorted(batch, key=lambda example: len(example['input_ids']), reverse=True)
    max_length = max(len(example['input_ids']) for example in sorted_batch)

    input_ids = torch.zeros((len(sorted_batch), max_length), dtype=torch.long)
    labels = []

    for row, example in enumerate(sorted_batch):
        ids = example['input_ids']
        input_ids[row, :len(ids)] = ids
        labels.append(example['label'])

    return {
        'input_ids': input_ids,
        'label': torch.stack(labels),
    }


def main():
    chapter08_dir = Path(__file__).resolve().parent
    train_dataset = torch.load(chapter08_dir / 'train_dataset.pt', map_location='cpu', weights_only=False)

    batch = collate(train_dataset[:4])

    print('入力事例:')
    for example in train_dataset[:4]:
        print(example)

    print('collate後:')
    print(batch)


if __name__ == '__main__':
    main()
