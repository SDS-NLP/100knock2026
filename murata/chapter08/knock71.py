import pandas as pd
import torch
import pickle

with open('vocab.pkl', 'rb') as f:
    token2id = pickle.load(f)['token2id']
    

def text_to_ids(text):
    return [token2id[w] for w in text.split() if w in token2id]

def load_sst(path):
    df = pd.read_csv(path, sep = '\t')
    out = []
    for _, row in df.iterrows():
        ids = text_to_ids(row['sentence'])
        
        if not ids:
            continue
        out.append({
            'text': row['sentence'],
            'label': torch.tensor([float(row['label'])]),
            'input_ids': torch.tensor(ids),
        })
        
    return out

train = load_sst('./SST-2/SST-2/train.tsv')
dev   = load_sst('./SST-2/SST-2/dev.tsv')
print(len(train), len(dev))
torch.save(train, 'sst_train.pt')
torch.save(dev,   'sst_dev.pt')