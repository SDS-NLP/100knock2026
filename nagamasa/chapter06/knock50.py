import gensim

# 50. 単語ベクトルの読み込みと表示
# 学習済み word2vec（GoogleNews, 300万語×300次元, バイナリ形式）を読み込み、
# "United States"（内部表現は "United_States"）のベクトルを表示する。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"

# word2vec バイナリ形式を読み込む
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
print("語彙数:", len(kv))          # 約300万語
print("次元数:", kv.vector_size)   # 300

# "United States" は語彙になく、"United_States" として登録されている
print("'United States' in vocab:", "United States" in kv)
print("'United_States' in vocab:", "United_States" in kv)

# "United_States" のベクトルを取り出して表示（kv[語] → 300次元の numpy 配列）
v = kv["United_States"]
print("United States のベクトル shape:", v.shape)  # (300,)
print("United States のベクトル:", v)
