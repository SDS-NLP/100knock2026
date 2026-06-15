from sklearn.cluster import KMeans

from knock50 import load_model


CAPITAL_SECTIONS = {
    ': capital-common-countries',
    ': capital-world',
}


def extract_countries(path='data/questions-words.txt'):
    countries = set()
    section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                section = line
                continue

            if section in CAPITAL_SECTIONS and line:
                _, country1, _, country2 = line.split()
                countries.add(country1)
                countries.add(country2)

    return sorted(countries)


def country_vectors(model, countries):
    words = [country for country in countries if country in model]
    vectors = [model[country] for country in words]
    return words, vectors


def kmeans_countries(model, n_clusters=5):
    countries = extract_countries()
    words, vectors = country_vectors(model, countries)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(vectors)
    return words, labels


if __name__ == '__main__':
    model = load_model()
    words, labels = kmeans_countries(model)

    for cluster_id in range(5):
        cluster = [
            word for word, label in zip(words, labels)
            if label == cluster_id
        ]
        print(f'cluster {cluster_id}: {", ".join(cluster)}')
