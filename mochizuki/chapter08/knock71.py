import csv
import json
import zipfile
import torch

SST_ZIP = 'SST-2.zip'

def load_token2id():
    with open('token2id.json') as f:
        return json.load(f)

def read_tsv(zf, name):
    with zf.open(name) as f:
        lines = f.read().decode('utf-8').splitlines()
    reader = csv.reader(lines, delimiter='\t')
    next(reader)
    return [(row[0].strip(), int(row[1])) for row in reader]

def make_examples(rows, token2id):
    examples = []
    for text, label in rows:
        ids = [token2id[t] for t in text.split() if t in token2id]
        if not ids:
            continue
        examples.append({
            'text': text,
            'label': torch.tensor([float(label)]),
            'input_ids': torch.tensor(ids),
        })
    return examples

def main():
    token2id = load_token2id()
    with zipfile.ZipFile(SST_ZIP) as zf:
        train_rows = read_tsv(zf, 'SST-2/train.tsv')
        dev_rows = read_tsv(zf, 'SST-2/dev.tsv')

    train = make_examples(train_rows, token2id)
    dev = make_examples(dev_rows, token2id)

    print(f'train size: {len(train)} (dropped {len(train_rows) - len(train)})')
    print(f'dev size:   {len(dev)} (dropped {len(dev_rows) - len(dev)})')
    print(f'\nexample: {train[1]}')

    torch.save({'train': train, 'dev': dev}, 'dataset.pt')
    print('\nsaved: dataset.pt')

if __name__ == '__main__':
    main()
