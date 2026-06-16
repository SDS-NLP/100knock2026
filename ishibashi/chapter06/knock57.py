import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

def run_kmeans_clustering():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'
    file = './chapter06/countries.txt'

    countries = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            countries.append(line.strip())

    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    valid_countries = []
    country_vectors = []
    for country in countries:
        if country in model:
            valid_countries.append(country)
            country_vectors.append(model[country])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(country_vectors)

    clusters = {i: [] for i in range(5)}
    for country, label in zip(valid_countries, kmeans.labels_):
        clusters[label].append(country)

    for i in range(5):
        print(f"Cluster_{i+1}:")
        print(", ".join(clusters[i]))

if __name__ == "__main__":
    run_kmeans_clustering()