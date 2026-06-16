import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import knock57

def main():
    Z = linkage(knock57.country_vectors, method='ward', metric = "euclidean")
    plt.figure(figsize=(12, 8))
    dendrogram(Z, labels = knock57.countries, leaf_rotation=90)
    plt.tight_layout()
    plt.savefig('knock58.png')


if __name__ == '__main__':
    main()

