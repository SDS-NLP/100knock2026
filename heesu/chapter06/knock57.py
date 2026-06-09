from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Extract country names from capital sections
countries = set()
capital_sections = {'capital-common-countries', 'capital-world'}
section = None

with open('questions-words.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            section = line[2:]
            continue
        if section not in capital_sections:
            continue
        parts = line.split()
        # format: capital1 country1 capital2 country2
        countries.update([parts[1], parts[3]])

# Keep only countries present in the model
countries = [c for c in sorted(countries) if c in model]

vectors = np.array([model[c] for c in countries])

kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
labels = kmeans.fit_predict(vectors)

for cluster_id in range(5):
    members = [countries[i] for i, l in enumerate(labels) if l == cluster_id]
    print(f"Cluster {cluster_id}: {', '.join(members)}")
