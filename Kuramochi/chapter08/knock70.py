import numpy as np
from gensim.models import KeyedVectors
import torch

def Word2Vectors(file_path: str):

    model     = KeyedVectors.load_word2vec_format(file_path, binary=True, limit=100000)
    d_emb     = model.vector_size 

    # 2. 対応付け(双方向)用の辞書を初期化し、<PAD>をインデックス0として登録
    word2id = {'<PAD>': 0}
    id2word = {0: '<PAD>'}

    vectors = [np.zeros(d_emb)]

    # 4. 学習済みモデルの単語とベクトルを順に追加していく
    for i, word in enumerate(model.index_to_key):

        token_id = i + 1
        
        # 双方向の対応付けを保持
        word2id[word]     = token_id
        id2word[token_id] = word

        vectors.append(model[word])

    # 5. 最後にリストをNumPy配列に変換して、単語埋め込み行列 E を作成
    E = torch.tensor(np.array(vectors), dtype=torch.float32)

    return word2id, id2word, E

def word2vec(word: str, word2id: dict, E)->int:
    id = word2id[word]
    return E[id]

def main(file_path: str, word: str)->list:
    word2id, id2word, E = Word2Vectors(file_path)
    word_vector         = word2vec(word, word2id, E)
    print(f"{word}: {word_vector[:10]}")

if __name__ == "__main__":
    file_path = "GoogleNews-vectors-negative300.bin"
    word      = "adult"
    main(file_path, word)