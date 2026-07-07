import torch
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)
token2id = {"<PAD>": 0}
token2id.update({tok: i + 1 for i, tok in enumerate(wv.index_to_key)})

def load_dataset(path, token2id):
    dataset = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            text, label = line.rstrip("\n").split("\t")
            ids = [token2id[tok] for tok in text.split() if tok in token2id]
            if len(ids) == 0:
                continue
            dataset.append({
                "text": text,
                "label": torch.tensor([float(label)]),
                "input_ids": torch.tensor(ids),
            })
    return dataset

train = load_dataset("SST-2/train.tsv", token2id)
dev = load_dataset("SST-2/dev.tsv", token2id)

print(f"train: {len(train)} 件, dev: {len(dev)} 件")
print(train[0])