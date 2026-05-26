import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)
score = model.similarity("United_States", "U.S.")
print(score)
