import pandas as pd
import torch
from transformers import BertTokenizer

MODEL = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL)

def load_sst(path):
    df = pd.read_csv(path, sep='\t')
    out = []
    for _, row in df.iterrows():
        text = str(row['sentence']).strip()
        tokens = tokenizer.tokenize(text)
        if not tokens:
            continue
        out.append({
            'text': text,
            'tokens': tokens,
            'input_ids': torch.tensor(tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + tokens + ['[SEP]'])),
            'label': torch.tensor(int(row['label'])),
        })
    return out

if __name__ == '__main__':
    train = load_sst('./SST-2/SST-2/train.tsv')
    dev   = load_sst('./SST-2/SST-2/dev.tsv')
    print(len(train), len(dev))
    for ex in train[:3]:
        print(ex['label'].item(), ex['tokens'])
    torch.save(train, 'sst_bert_train.pt')
    torch.save(dev,   'sst_bert_dev.pt')
