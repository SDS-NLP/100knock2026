import torch
from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0   # bert-base-uncased の [PAD]

def collate(batch):
    input_ids = pad_sequence(
        [ex['input_ids'] for ex in batch],
        batch_first=True, padding_value=PAD_ID
    )
    attention_mask = (input_ids != PAD_ID).long()
    labels = torch.stack([ex['label'] for ex in batch])
    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels}

if __name__ == '__main__':
    train = torch.load('sst_bert_train.pt')
    out = collate(train[:4])
    print(out['input_ids'])
    print(out['attention_mask'])
    print(out['label'])
