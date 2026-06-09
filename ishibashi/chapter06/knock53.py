from gensim.models import KeyedVectors

def calculate_analogy():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    positives = ['Spain', 'Athens']
    negatives = ['Madrid']

    for word in positives + negatives:
        if word not in model:
            print(f'"{word}"がモデルに存在しません')
            return
    
    print("計算式: Spain - Madrid + Athens")

    similar_words = model.most_similar(positive=positives, negative=negatives, topn=10)

    for i, (sim_word, similarity) in enumerate(similar_words, 1):
        print(f"{i}位: {sim_word}\n類似度: {similarity:.4f}")

if __name__ == "__main__":
    calculate_analogy()