import gensim

# 51. 単語の類似度
# "United States"（内部表現 "United_States"）と "U.S." のコサイン類似度を計算する。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# コサイン類似度を計算（kv.similarity は L2 正規化したベクトル同士の内積を返す）
print("similarity(United_States, U.S.):", kv.similarity("United_States", "U.S."))


"""
similarity(United_States, U.S.): 0.73107743
"""
