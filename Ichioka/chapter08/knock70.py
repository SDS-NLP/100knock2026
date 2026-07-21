"""
事前学習済み単語ベクトル（Google News, word2vec形式）から
単語埋め込み行列と語彙辞書を構築するスクリプト.

行列の形状は (V+1, D):
  - 行0  : <PAD> トークン用ゼロベクトル（予約済み）
  - 行1~ : 事前学習済み単語ベクトル（V 語彙分）
"""

import numpy as np
from gensim.models import KeyedVectors


def build_embedding_matrix(
    model_path: str,
) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    """
    事前学習済み単語ベクトルから埋め込み行列と語彙辞書を構築する.

    行インデックス0はパディングトークン <PAD> 用のゼロベクトルとして予約し,
    1行目以降に事前学習済み単語ベクトルを格納する.

    Parameters
    ----------
    model_path : str
        Word2Vec バイナリ形式（.bin.gz）のファイルパス.

    Returns
    -------
    embedding_matrix : np.ndarray
        形状 (V+1, D) の単語埋め込み行列.
        embedding_matrix[0] はゼロベクトル（<PAD> 用）.
    word2id : dict[str, int]
        単語からトークンID（行インデックス）への対応辞書.
    id2word : dict[int, str]
        トークンIDから単語への対応辞書.
    """
    # 事前学習済みモデルの読み込み
    print("モデルを読み込み中...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("モデルの読み込み完了.")

    # 語彙数 V と埋め込み次元数 D の取得
    V: int = len(model.index_to_key)  # Google News なら 3,000,000
    D: int = model.vector_size         # Google News なら 300

    # 埋め込み行列を (V+1) × D のゼロ行列で初期化
    # 行0は <PAD> トークン用ゼロベクトルとして予約済み
    embedding_matrix = np.zeros((V + 1, D), dtype=np.float32)

    # 事前学習済み単語ベクトルを1行目以降に一括コピー
    # model.vectors の行順は model.index_to_key の語彙順と対応している
    embedding_matrix[1:] = model.vectors

    # 双方向マッピングの構築
    # トークンID=0 は <PAD> 用に予約し, 1 から V まで単語を割り当てる
    word2id: dict[str, int] = {"<PAD>": 0}
    id2word: dict[int, str] = {0: "<PAD>"}

    for token_id, word in enumerate(model.index_to_key, start=1):
        word2id[word] = token_id
        id2word[token_id] = word

    return embedding_matrix, word2id, id2word


if __name__ == "__main__":
    model_path = "GoogleNews-vectors-negative300.bin.gz"

    # 埋め込み行列と語彙辞書の構築
    embedding_matrix, word2id, id2word = build_embedding_matrix(model_path)

    # ---- 結果の確認 ----
    V_plus_1, D = embedding_matrix.shape
    V = V_plus_1 - 1

    print(f"語彙数  V          : {V:,}")
    print(f"次元数  D          : {D}")
    print(f"埋め込み行列の形状  : {embedding_matrix.shape}  （期待値: (3000001, 300)）")

    # 行0がゼロベクトルであることの検証
    assert np.all(embedding_matrix[0] == 0.0), "行0がゼロベクトルではありません"
    print("行0 (<PAD>) はゼロベクトルです ✓")

    # United_States の動作確認
    word = "United_States"
    if word in word2id:
        tid = word2id[word]
        print(f"'{word}' のトークンID        : {tid}")
        print(f"'{word}' の逆引き確認        : {id2word[tid]}")
        print(f"'{word}' のベクトル（先頭5次元）: {embedding_matrix[tid, :5]}")
    else:
        print(f"'{word}' は語彙に含まれていません")