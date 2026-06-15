from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np

model = KeyedVectors.load_word2vec_format(
    'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

countries = set()
target_categories = {"capital-common-countries", "capital-world", "currency"}
current = None
with open("100knock2026/questions-words.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith(":"):
            current = line[2:].strip()
            continue
        if current in target_categories:
            words = line.split()
            if "capital" in current:
                countries.add(words[1])
                countries.add(words[3])
            elif current == "currency":
                countries.add(words[0])
                countries.add(words[2])
countries = sorted([c for c in countries if c in model.key_to_index])
print(f"国数: {len(countries)}")
vectors = np.array([model[c] for c in countries])

km = KMeans(n_clusters=5, random_state=216, n_init=10)
labels = km.fit_predict(vectors)   

from collections import defaultdict        
clusters = defaultdict(list)
for c, l in zip(countries, labels):
    clusters[l].append(c)
for k, members in sorted(clusters.items()):
    print(f"\n--- Cluster {k} ({len(members)}カ国) ---")
    print(", ".join(members))