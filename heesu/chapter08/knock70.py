"""70. 単語埋め込みの読み込み

事前学習済み単語埋め込み (Google News, 300万語 x 300次元) を読み込み、
単語埋め込み行列 E (V x D) を作成する。
- E の先頭行 (index 0) は <PAD> トークン用にゼロベクトルで予約する。
- よって 2 行目以降に事前学習済みベクトルを読み込む。
- トークンID <-> 単語 の双方向の対応付けも同時に保持する。

全語彙を読み込むと E.shape == (3000001, 300) になる。
"""

import numpy as np
from gensim.models import KeyedVectors

EMBEDDING_PATH = "GoogleNews-vectors-negative300.bin"
PAD_TOKEN = "<PAD>"
PAD_ID = 0


def load_embeddings(path=EMBEDDING_PATH, limit=None):
    """事前学習済み単語埋め込みを読み込む。

    Args:
        path: word2vec バイナリファイルへのパス。
        limit: 語彙数の上限 (頻度上位 limit 語のみ読み込む)。None なら全語彙。

    Returns:
        embedding_matrix: np.ndarray, shape (V, D), dtype float32。行 0 はゼロベクトル。
        word_to_id: dict[str, int], 単語 -> トークンID。
        id_to_word: dict[int, str], トークンID -> 単語。
    """
    kv = KeyedVectors.load_word2vec_format(path, binary=True, limit=limit)
    vocab_size = len(kv) + 1  # +1 は <PAD> の分
    dim = kv.vector_size

    embedding_matrix = np.zeros((vocab_size, dim), dtype=np.float32)
    embedding_matrix[1:] = kv.vectors  # 行 0 はゼロベクトルのまま予約

    word_to_id = {PAD_TOKEN: PAD_ID}
    id_to_word = {PAD_ID: PAD_TOKEN}
    for word, idx in kv.key_to_index.items():
        token_id = idx + 1  # PAD の分だけ 1 ずらす
        word_to_id[word] = token_id
        id_to_word[token_id] = word

    return embedding_matrix, word_to_id, id_to_word


if __name__ == "__main__":
    embedding_matrix, word_to_id, id_to_word = load_embeddings()

    print("embedding matrix shape:", embedding_matrix.shape)
    print("dtype:", embedding_matrix.dtype)
    print("vocab size (incl. PAD):", len(word_to_id))
    print("row 0 is zero vector (PAD):", bool(np.all(embedding_matrix[0] == 0)))

    print("\n-- mapping examples --")
    print("word_to_id['<PAD>'] =", word_to_id[PAD_TOKEN])
    print("word_to_id['United_States'] =", word_to_id["United_States"])
    print("id_to_word[1] =", id_to_word[1])
    # 双方向の対応付けが一致することを確認
    w = "United_States"
    print(f"id_to_word[word_to_id['{w}']] == '{w}':",
          id_to_word[word_to_id[w]] == w)
