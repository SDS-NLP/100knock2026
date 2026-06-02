from knock50 import load_model

def most_similar(model, positive, negative=None, topn=10):
    return model.most_similar(positive, negative=negative, topn=topn)

if __name__ == '__main__':
    model = load_model()
    print("United states と類似する単語")
    for word, score in most_similar(model,  'United_States', topn=10):
        print(f'{word}: {score:.4f}')


