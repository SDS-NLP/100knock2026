from gensim.models import KeyedVectors
import torch
import pickle

EMBEDDING_PATH = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

kv = KeyedVectors.load_word2vec_format(
    EMBEDDING_PATH,
    binary=True,
    limit=None
)

d_emb = kv.vector_size

id_to_token = ["<PAD>"] + kv.index_to_key
token_to_id = {token: i for i, token in enumerate(id_to_token)}

E = torch.zeros((len(id_to_token), d_emb), dtype=torch.float32)
E[1:] = torch.from_numpy(kv.vectors)

torch.save(E, "embedding_matrix.pt")

with open("token_to_id.pkl", "wb") as f:
    pickle.dump(token_to_id, f)

with open("id_to_token.pkl", "wb") as f:
    pickle.dump(id_to_token, f)