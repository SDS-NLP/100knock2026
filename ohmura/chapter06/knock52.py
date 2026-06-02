from gensim.models import KeyedVectors

model_path = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

target_word = 'United_States'

similar_words = model.most_similar(target_word, topn=10)

print(f"「{target_word}」とコサイン類似度が高い単語トップ10:\n")
print(f"{'順位':<4} | {'単語':<15} | {'類似度'}")
print("-" * 35)

for i, (word, similarity) in enumerate(similar_words, 1):
    print(f"{i:<5} | {word:<15} | {similarity:.4f}")