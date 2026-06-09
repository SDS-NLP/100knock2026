from gensim.models import KeyedVectors

def load_and_display_vector():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    word = 'United_States'
    if word in model:
        vector = model[word]

        print(f'単語: "{word}"')
        print(f"ベクトルの次元数: {vector.shape}次元")

        print(f"単語ベクトル:\n{vector}")

    else:
        print(f'単語"{word}"はモデルに存在しません')

if __name__ == "__main__":
    load_and_display_vector()