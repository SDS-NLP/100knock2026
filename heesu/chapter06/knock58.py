from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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
        countries.update([parts[1], parts[3]])

countries = [c for c in sorted(countries) if c in model]
vectors = np.array([model[c] for c in countries])

Z = linkage(vectors, method='ward')

fig, ax = plt.subplots(figsize=(12, len(countries) * 0.3 + 2))
dendrogram(Z, labels=countries, orientation='right', ax=ax)
ax.set_title('Hierarchical Clustering of Country Vectors (Ward)')
plt.tight_layout()
plt.savefig('knock58_dendrogram.png', dpi=100)
print("Saved: knock58_dendrogram.png")
