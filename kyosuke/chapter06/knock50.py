import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'

model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
word = "United_States"
vec = model[word]
print(vec[:10])