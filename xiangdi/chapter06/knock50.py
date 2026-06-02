from gensim.models import KeyedVectors

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(model_path, binary=True)

vector = model["United_States"]

print(vector)