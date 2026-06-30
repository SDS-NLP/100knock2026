import numpy as np
from gensim.models import KeyedVectors

wv = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin.gz", binary=True
)

vocab = wv.index_to_key
d_emb = wv.vector_size
V = len(vocab) + 1

E = np.zeros((V, d_emb), dtype=np.float32)
E[1:] = wv.vectors

token2id = {"<PAD>": 0}
token2id.update({tok: i + 1 for i, tok in enumerate(vocab)})
id2token = {i: tok for tok, i in token2id.items()}

print(E.shape)
print(np.all(E[0] == 0))
print(token2id["dog"], id2token[token2id["dog"]])