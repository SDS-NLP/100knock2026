from gensim.models import KeyedVectors

file = "data/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

print(model["United_States"])
