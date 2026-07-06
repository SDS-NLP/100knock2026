import torch
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    batch = sorted(batch, key=lambda x: len(x['input_ids']), reverse=True)
    input_ids = pad_sequence(
        [ex['input_ids'] for ex in batch],
        batch_first=True, padding_value=0
    )
    labels = torch.stack([ex['label'] for ex in batch])
    return {'input_ids': input_ids, 'label': labels}

if __name__ == '__main__':
    train = torch.load('sst_train.pt')
    out = collate(train[:4])
    print(out['input_ids'])
    print(out['label'])