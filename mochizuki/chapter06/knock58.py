import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from knock57 import countries, vectors

if __name__ == '__main__':
    Z = linkage(vectors, method='ward')
    plt.figure(figsize=(12, 8))
    dendrogram(Z, labels=countries, leaf_rotation=90)
    plt.tight_layout()
    plt.savefig('knock58.png')
