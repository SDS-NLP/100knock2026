from gensim.models import KeyedVectors

file_path = 'GoogleNews-vectors-negative300.bin'

print("巨大なモデルを読み込んでいます...")
model = KeyedVectors.load_word2vec_format(file_path, binary=True)

similar_words = model.most_similar('United_States', topn=10)

print("--- 「United_States」と似ている単語トップ10 ---")
for rank, (word, similarity) in enumerate(similar_words, 1):
    print(f"{rank}位: {word} (類似度: {similarity:.4f})")
