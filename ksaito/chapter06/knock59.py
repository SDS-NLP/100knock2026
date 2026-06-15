import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from knock50 import load_model
from knock57 import country_vectors, extract_countries


def tsne_countries(model):
    countries = extract_countries()
    words, vectors = country_vectors(model, countries)
    tsne = TSNE(
        n_components=2,
        random_state=0,
        init='pca',
        learning_rate='auto',
        perplexity=30,
    )
    embedded = tsne.fit_transform(vectors)
    return words, embedded


if __name__ == '__main__':
    model = load_model()
    words, embedded = tsne_countries(model)

    plt.figure(figsize=(14, 10))
    plt.scatter(embedded[:, 0], embedded[:, 1])
    for word, point in zip(words, embedded):
        plt.annotate(word, (point[0], point[1]), fontsize=8)

    plt.tight_layout()
    plt.savefig('data/country_tsne.png')
    plt.show()
