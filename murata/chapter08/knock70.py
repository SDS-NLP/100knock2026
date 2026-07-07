import gensim
import torch
import pickle

MODEL_PATH = "./GoogleNews-vectors-negative300.bin.gz"
N = 500000
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True, limit=N)

token2id = {'<PAD>': 0}
id2token = {0: '<PAD>'}

for i, w in enumerate(kv.key_to_index.keys(), start=1):
    token2id[w] = i
    id2token[i] = w
    
V, d = len(token2id), kv.vector_size
E = torch.zeros(V, d)
for w, i in token2id.items():
    if i==0:
        continue
    E[i] = torch.from_numpy(kv[w].copy())

print(f'E.shape = {E.shape}')   # (3000001, 300)
torch.save(E, 'embedding_matrix.pt')
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'token2id': token2id, 'id2token': id2token}, f)
