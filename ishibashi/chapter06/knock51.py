from gensim.models import KeyedVectors

def calculate_similarity():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    word1 = 'United_States'
    word2 = 'U.S.'
    if word1 in model and word2 in model:
        similarity = model.similarity(word1, word2)

        print(f'"{word1}"と"{word2}"のコサイン類似度: {similarity}')
    else:
        print("指定された単語がモデルに存在しません")

if __name__ == "__main__":
    calculate_similarity()