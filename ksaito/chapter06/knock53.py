from knock50 import load_model
from knock52 import most_similar

if __name__ == '__main__':
    model = load_model()
    print("Spain と Athens から Madrid を引いたときに類似する単語")
    for word, score in most_similar(model, positive=['Spain', 'Athens'], negative=['Madrid'], topn=10):
        print(f'{word}: {score:.4f}')
