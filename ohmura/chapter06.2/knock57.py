from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

countries = set()
with open('questions-words.txt', 'r') as f:
    is_target = False
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            is_target = line in [': capital-common-countries', ': capital-world']
            continue
        
        if is_target:
            words = line.split()
            if len(words) == 4:
                countries.add(words[1])
                countries.add(words[3])

countries = list(countries)

country_vecs = []
valid_countries = []
for country in countries:
    if country in model:
        country_vecs.append(model[country])
        valid_countries.append(country)

kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
kmeans.fit(country_vecs)

clusters = {i: [] for i in range(5)}
for i, label in enumerate(kmeans.labels_):
    clusters[label].append(valid_countries[i])

for i in range(5):
    print(f"=== クラスタ {i} ===")
    print(", ".join(clusters[i]))
    print()