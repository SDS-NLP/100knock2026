import torch
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors

def load_embeddings(limit=100000):
    model_path = api.load("word2vec-google-news-300", return_path=True)
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=limit)
    
    vocab_size = len(model.key_to_index) + 1
    embed_dim = model.vector_size
    
    weights = np.zeros((vocab_size, embed_dim))
    word2id = {'<PAD>': 0}
    id2word = {0: '<PAD>'}
    
    for i, word in enumerate(model.index_to_key, 1):
        word2id[word] = i
        id2word[i] = word
        weights[i] = model[word]
        
    embedding_tensor = torch.tensor(weights, dtype=torch.float32)
    
    return embedding_tensor, word2id, id2word

if __name__ == '__main__':
    E, word2id, id2word = load_embeddings(limit=100000)
    
    print(f"単語埋め込み行列の形状: {E.shape}")
    print(f"ID=0 (<PAD>) のベクトル先頭5要素: {E[0][:5]}")
    print(f"ID=1 の単語: {id2word[1]}")