import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)

V, d_emb = model.vectors.shape
matrix = np.zeros((V + 1, d_emb))

id_to_key = {0: "<PAD>"}
key_to_id = {"<PAD>": 0}

for key, id in model.key_to_index.items():
    id += 1
    id_to_key[id] = key
    key_to_id[key] = id
    matrix[id, :] = model[key]

print(f"単語埋め込み行列の形状: {matrix.shape}")

# result
# 単語埋め込み行列の形状: (3000001, 300)
