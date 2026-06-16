from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format(f'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz', binary=True)

print(model['United_States'])