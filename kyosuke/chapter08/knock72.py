import torch
import torch.nn as nn
import numpy as np
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)
V = len(wv.index_to_key) + 1
d_emb = wv.vector_size

E = np.zeros((V, d_emb), dtype=np.float32)
E[1:] = wv.vectors
E = torch.from_numpy(E)

class BoWClassifier(nn.Module): #nn.Moduleを親クラスとして継承
    def __init__(self, embedding_weights, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained( #IDを渡すとベクトルを返す
            embedding_weights,
            freeze=False,
            padding_idx=padding_idx,
        )
        d_emb = embedding_weights.size(1) #300次元
        self.fc = nn.Linear(d_emb, 1) #300個の値から1つの値を出力

    def forward(self, input_ids): 
        emb = self.embedding(input_ids)
        feat = emb.mean(dim=1)
        logit = self.fc(feat)
        return logit

model = BoWClassifier(E)
example_ids = torch.tensor([[3475, 87, 15888, 90, 27695, 42637]])
logit = model(example_ids) #model.__call__(example_ids)⇒model.forward(example_ids)
prob = torch.sigmoid(logit)
print("logit:", logit)
print("ポジティブ確率:", prob.item())