import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

result = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
for i, (word, score) in enumerate(result, 1):
    print(f"{i:2d}位: {word} (類似度: {score:.4f})")