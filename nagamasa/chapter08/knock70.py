import numpy as np
from gensim.models import KeyedVectors

# 70. 単語埋め込みの読み込み
# 学習済み単語ベクトル(GoogleNews, 300次元)から |V|×demb の埋め込み行列 E を作る。
# E[0] は PAD 用にゼロベクトルを予約する → 実単語の id は 1 から始まる(=word2vec の並びを 1 ずらす)。
# トークンID ↔ 単語 の双方向対応も同時に保持する。

W2V_PATH = "../chapter06/GoogleNews-vectors-negative300.bin"
PAD = "<PAD>"

# bin は頻度降順で並ぶので limit=N は頻出上位 N 語のロードを意味する(全件は不要)。
# 最終的な N は 71 で SST の被覆率を測ってから決める。ここでは暫定で 10 万語。
LIMIT = 100000


def build_embedding_matrix(path=W2V_PATH, limit=LIMIT):
    kv = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
    vocab = kv.index_to_key   # 頻度降順の単語リスト(長さ = limit)
    demb = kv.vector_size     # 300

    # 0 行目を PAD のゼロベクトルにし、1 行目以降に学習済みベクトルを詰める。
    # kv.vectors は index_to_key と同じ並びの (limit, demb) 配列なので、
    # 1 行ずらして一括コピーすれば word i → 行 i+1 が成立する。
    E = np.zeros((len(vocab) + 1, demb), dtype=np.float32)
    E[1:] = kv.vectors

    # 双方向対応。PAD を id=0 に固定し、実単語は 1..limit。
    word2id = {PAD: 0}
    for i, word in enumerate(vocab):
        word2id[word] = i + 1
    id2word = {i: w for w, i in word2id.items()}

    # 1ずらしの実検証: word を引いた id の行に、その word のベクトルが載っているか。
    # kv が手元にある関数内で突き合わせる(id2word/word2id だけの比較は恒真で検証にならない)。
    assert all(np.array_equal(E[word2id[w]], kv[w]) for w in vocab[:5]), "1ずらしが壊れている"

    return E, word2id, id2word


if __name__ == "__main__":
    E, word2id, id2word = build_embedding_matrix()

    print("E shape:", E.shape)                       # (limit+1, 300)
    print("id2word[0]:", id2word[0])                 # <PAD>
    print("id2word[1]:", id2word[1])                 # 最頻語
    print("E[0] is zero (PAD):", not E[0].any())     # True: PAD 行はゼロ
    print("E[1] is non-zero:", bool(E[1].any()))     # True: 実ベクトルは行1から
