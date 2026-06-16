import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

from knock50 import load_model
from knock57 import country_vectors, extract_countries


def ward_countries(model):
    countries = extract_countries()
    words, vectors = country_vectors(model, countries)
    return words, linkage(vectors, method='ward')


if __name__ == '__main__':
    model = load_model()
    words, linked = ward_countries(model)

    plt.figure(figsize=(16, 8))
    dendrogram(linked, labels=words, leaf_font_size=8)
    plt.tight_layout()
    plt.savefig('data/country_dendrogram.png')
    plt.show()
