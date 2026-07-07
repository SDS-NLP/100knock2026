import numpy as np
from gensim.models import KeyedVectors


def load_pretrained_embeddings(path, limit=None):
    return KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)


def create_embedding_matrix(pretrained_embeddings):
    V = len(pretrained_embeddings) + 1
    d_emb = pretrained_embeddings.vector_size

    E = np.zeros((V, d_emb), dtype=pretrained_embeddings.vectors.dtype)
    token_to_id = {"<PAD>": 0}
    id_to_token = {0: "<PAD>"}

    for id, token in enumerate(pretrained_embeddings.index_to_key, 1):
        E[id, :] = pretrained_embeddings[token]
        token_to_id[token] = id
        id_to_token[id] = token

    return E, token_to_id, id_to_token


def main():
    path = "GoogleNews-vectors-negative300.bin.gz"
    pretrained_embeddings = load_pretrained_embeddings(path, limit=100000)
    E, token_to_id, id_to_token = create_embedding_matrix(pretrained_embeddings)

    V, d_emb = E.shape
    print(f"|V| = {V}, d_emb = {d_emb}")
    print(f"E.dtype = {E.dtype}")
    print(f"E[0]の最初の5行： {E[0][:5]}")
    print("-"*60)

    print("token_to_id, id_to_token の確認")
    sample = "dog"
    print(f"token_to_id['{sample}'] = {token_to_id[sample]}")
    print(f"id_to_token[{token_to_id[sample]}] = {id_to_token[token_to_id[sample]]}")


if __name__ == "__main__":
    main()
