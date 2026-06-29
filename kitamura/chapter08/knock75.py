import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

class TextClassifier(nn.Module):
    def __init__(self, embedding_tensor):
        super().__init__() #親クラスの設定の引き継ぎ

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            padding_idx=0,
            freeze=True
        )
        embed_dim = embedding_tensor.size(1)
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)          
        sum_embeddings = (embedded * mask).sum(dim=1)  
        valid_lengths = mask.sum(dim=1).clamp(min=1)   
        avg_embedded = sum_embeddings / valid_lengths    
        out = self.linear(avg_embedded)
        return out


def collate_fn(batch):
    sorted_batch = sorted(batch, key=lambda x:len(x["input_ids"]), reverse=True)
    input_ids = []
    labels = []
    
    for item in sorted_batch:
        input_ids.append(item["input_ids"])
        labels.append(item["label"])
    
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return {'input_ids': padded_ids, 'labels': labels}

# 問題文で与えられた4つのサンプルデータ
sample_batch = [
    {'text': 'hide new secretions from the parental units', 'label': torch.tensor([0.]), 'input_ids': torch.tensor([5785, 66, 113845, 18, 12, 15095, 1594])},
    {'text': 'contains no wit , only labored gags', 'label': torch.tensor([0.]), 'input_ids': torch.tensor([3475, 87, 15888, 90, 27695, 42637])},
    {'text': 'that loves its characters and communicates something rather beautiful about human nature', 'label': torch.tensor([1.]), 'input_ids': torch.tensor([4, 5053, 45, 3305, 31647, 348, 904, 2815, 47, 1276, 1964])},
    {'text': 'remains utterly satisfied to remain the same throughout', 'label': torch.tensor([0.]), 'input_ids': torch.tensor([987, 14528, 4941, 873, 12, 208, 898])}
]

# collate_fn を通して結果を確認
result = collate_fn(sample_batch)

print("【パディングとソートの結果】")
print("input_ids:\n", result['input_ids'])
print("\nlabels:\n", result['labels'])

"""input_ids:
 tensor([[     4,   5053,     45,   3305,  31647,    348,    904,   2815,     47,
           1276,   1964],
        [  5785,     66, 113845,     18,     12,  15095,   1594,      0,      0,
              0,      0],
        [   987,  14528,   4941,    873,     12,    208,    898,      0,      0,
              0,      0],
        [  3475,     87,  15888,     90,  27695,  42637,      0,      0,      0,
              0,      0]])

labels:
 tensor([[1.],
        [0.],
        [0.],
        [0.]])"""