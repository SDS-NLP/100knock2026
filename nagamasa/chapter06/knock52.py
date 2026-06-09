import gensim

# 52. 類似度の高い単語10件
# "United States"（内部表現 "United_States"）とコサイン類似度が高い 10 語と類似度を出力する。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# 上位10語を取得（most_similar はソート済みの (語, 類似度) リストを返す）
print("most_similar(United_States):")
for word, score in kv.most_similar("United_States", topn=10):
    print(f"  {word}\t{score:.4f}")


"""
most_similar(United_States):
Unites_States	0.7877
Untied_States	0.7541
United_Sates	0.7401
U.S.	0.7311
theUnited_States	0.6404
America	0.6178
UnitedStates	0.6167
Europe	0.6133
countries	0.6045
Canada	0.6019
"""
