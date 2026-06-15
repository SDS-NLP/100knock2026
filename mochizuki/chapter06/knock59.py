import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from knock57 import countries, vectors

if __name__ == '__main__':
    tsne = TSNE(n_components=2, random_state=0)
    coords = tsne.fit_transform(vectors)

    plt.figure(figsize=(14, 10))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, country in enumerate(countries):
        plt.annotate(country, (coords[i, 0], coords[i, 1]))
    plt.tight_layout()
    plt.savefig('knock59.png')
