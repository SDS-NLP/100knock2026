from pathlib import Path
import csv

import joblib
import torch


def text_to_input_ids(text, word2id):
    return [word2id[token] for token in text.strip().split() if token in word2id]


def load_sst2_dataset(tsv_path, word2id):
    dataset = []

    with tsv_path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        for row in reader:
            text = row['sentence'].strip()
            label = float(row['label'])
            input_ids = text_to_input_ids(text, word2id)

            if not input_ids:
                continue

            dataset.append({
                'text': text,
                'label': torch.tensor([label], dtype=torch.float32),
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
            })

    return dataset


def main():
    chapter08_dir = Path(__file__).resolve().parent
    data_dir = chapter08_dir / 'SST-2'
    word2id_path = chapter08_dir / 'word2id.joblib'

    if not word2id_path.exists():
        raise FileNotFoundError(f'{word2id_path} が見つかりません。先に knock70.py を実行してください。')

    word2id = joblib.load(word2id_path)

    train_dataset = load_sst2_dataset(data_dir / 'train.tsv', word2id)
    dev_dataset = load_sst2_dataset(data_dir / 'dev.tsv', word2id)

    torch.save(train_dataset, chapter08_dir / 'train_dataset.pt')
    torch.save(dev_dataset, chapter08_dir / 'dev_dataset.pt')

    print(f'訓練セット: {len(train_dataset)} 件')
    print(f'開発セット: {len(dev_dataset)} 件')
    print('訓練セットの最初の事例:')
    print(train_dataset[0])
    print('開発セットの最初の事例:')
    print(dev_dataset[0])


if __name__ == '__main__':
    main()
