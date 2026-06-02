import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

word = "United_States"
similar_words = model.most_similar(word, topn=10)
print(similar_words)
for i, (sim_word, score) in enumerate(similar_words, 1):
    print(f"{i:2d}位: {sim_word} (類似度: {score:.4f})")