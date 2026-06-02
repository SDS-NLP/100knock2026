from gensim.models import KeyedVectors

file_path = 'GoogleNews-vectors-negative300.bin'

print("巨大なモデルを読み込んでいます...")
model = KeyedVectors.load_word2vec_format(file_path, binary=True)
print('読み込みが完了しました.')

print('指定のベクトルの計算をし、それに基づく類似度を計算しています...')
similar_words = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)

print("--- 「Spain - Madrid + Athens」の計算結果 トップ10 ---")
for rank, (word, similarity) in enumerate(similar_words, 1):
    print(f"{rank}位: {word} (類似度: {similarity:.4f})")