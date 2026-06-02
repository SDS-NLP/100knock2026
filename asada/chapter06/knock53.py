import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)
similar_words = model.most_similar(
    positive=["Spain", "Athens"], negative=["Madrid"], topn=10
)
print(similar_words)
