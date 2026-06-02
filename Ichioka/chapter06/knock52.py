from gensim.models import KeyedVectors

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

word = "United_States"

# "United_States" とコサイン類似度が高い上位10語を取得
similar_words = model.most_similar(word, topn=10)

for similar_word, similarity in similar_words:
    print(f"{similar_word}\t{similarity}")