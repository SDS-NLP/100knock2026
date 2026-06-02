from gensim.models import KeyedVectors

file_path = 'GoogleNews-vectors-negative300.bin'

print("巨大なモデルを読み込んでいます...")
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

similarity = model.similarity('United_States', 'U.S.')
print(f"「United_States」と「U.S.」のコサイン類似度: {similarity}")

# word_1   = 'United_States'
# word_2   = 'U.S.'
# vector_1 = model[word_1]
# vector_2 = model[word_2]

