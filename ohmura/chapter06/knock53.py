from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('../chapter06/GoogleNews-vectors-negative300.bin', binary=True)

answers = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)

for word, similarity in answers:
    print(f"{word}: {similarity:.4f}")