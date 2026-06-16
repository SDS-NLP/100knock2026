from gensim.models import KeyedVectors

def find_similar_words():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    word = 'United_States'
    if word in model:
        similar_words = model.most_similar(word, topn=10)

        for i, (sim_word, similarity) in enumerate(similar_words, 1):
            print(f"{i}位: {sim_word}\n類似度: {similarity:.4f}")
    else:
        print("指定された単語がモデルに存在しません")

if __name__ == "__main__":
    find_similar_words()