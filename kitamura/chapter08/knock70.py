import torch
import numpy as np
from gensim.models import KeyedVectors
import gensim.downloader as api

# Google Newsデータセットで学習済みのWord2Vecモデル（300次元）をロード
model = api.load('word2vec-google-news-300')

def embedding(w2v, vocab_limit):
    embed_dim = w2v.vector_size  # 単語の次元数
    vocab_size = len(w2v) + 1  # 語彙の総数（読み込んだ10万単語＋１（PAD)

    word2id = {"<PAD>":0}  # 単語とIDのdict (ID:0の単語は<PAD>)
    id2word = ["<PAD>"]  # 単語のlist

    weights = np.zeros((vocab_size, embed_dim), dtype=np.float32) # 単語のベクトルを格納する行列

    for i, word in enumerate(w2v.index_to_key):
        idx = i+1 # 0はPADなので飛ばす

        word2id[word] = idx
        id2word.append(word)
        weights[idx] = w2v[word] # weightのidx行目にその単語のベクトルを入れる

    embedding_tensor = torch.from_numpy(weights) # 行列をテンソルに変換する

    return embedding_tensor, word2id, id2word


embedding_tensor, word2id, id2word = embedding(model, 100000)
print(f"行列の形状：{embedding_tensor.shape}")
print(f"先頭ベクトル(<PAD>): {embedding_tensor[0][:5]}") # すべて0.0になっているか確認
print(f"ID 1 の単語: {id2word[1]}")
print(f"'good'のID: {word2id.get('good', '辞書にありません')}")

save_data = {
    "embedding_tensor" : embedding_tensor,
    "word2id" : word2id,
    "id2word" : id2word
}

file_name = "word2vec_vocab_embedding.pt"
torch.save(save_data, file_name)
print("save")

"""
行列の形状：torch.Size([3000001, 300])
先頭ベクトル(<PAD>): tensor([0., 0., 0., 0., 0.]) ...
ID 1 の単語: </s>
'good'のID: 128
"""