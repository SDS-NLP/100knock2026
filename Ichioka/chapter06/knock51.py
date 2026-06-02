from gensim.models import KeyedVectors

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

word1 = "United_States"
word2 = "U.S."

similarity = model.similarity(word1, word2)

print(similarity)