import gensim

# 53. 加法構成性によるアナロジー
# vec(Spain) - vec(Madrid) + vec(Athens) に類似度が高い 10 語と類似度を出力せよ。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# vec(Spain) - vec(Madrid) + vec(Athens) に近い上位10語を取得する
# （足す語=positive, 引く語=negative。most_similar は入力語を結果から自動除外する）
print("most_similar(Spain - Madrid + Athens):")
for word, score in kv.most_similar(
    positive=["Spain", "Athens"], negative=["Madrid"], topn=10
):
    print(f"  {word}\t{score:.4f}")

"""
most_similar(Spain - Madrid + Athens):
Greece	0.6898
Aristeidis_Grigoriadis	0.5607
Ioannis_Drymonakos	0.5553
Greeks	0.5451
Ioannis_Christou	0.5401
Hrysopiyi_Devetzi	0.5248
Heraklio	0.5208
Athens_Greece	0.5169
Lithuania	0.5167
Iraklion	0.5147
"""
