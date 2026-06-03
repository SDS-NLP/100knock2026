from gensim.models import KeyedVectors

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

# "United States" は内部的には "United_States" と表現されているらしい
word = "United_States"
vector = model[word]

print(vector)