from gensim.models import KeyedVectors

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(model_path, binary=True)

results = model.most_similar("United_States", topn=10)

for word, similarity in results:
    print(word, similarity)