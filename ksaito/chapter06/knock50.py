from gensim.models import KeyedVectors

def load_model(path='data/GoogleNews-vectors-negative300.bin.gz'):
    return KeyedVectors.load_word2vec_format(path, binary=True)


if __name__ == '__main__':
    model = load_model()
    vector = model['United_States']
    print(vector)
