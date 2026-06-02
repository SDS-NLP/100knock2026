from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

similar_words = model.most_similar('United_States', topn=10)
for word, score in similar_words:
    print(f'{word}\t{score:.4f}')
