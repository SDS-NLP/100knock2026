from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
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

tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(countries) - 1))
coords = tsne.fit_transform(vectors)

fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(coords[:, 0], coords[:, 1], s=20)
for i, name in enumerate(countries):
    ax.annotate(name, (coords[i, 0], coords[i, 1]), fontsize=7, textcoords='offset points', xytext=(4, 2))
ax.set_title('t-SNE of Country Word Vectors')
ax.axis('off')
plt.tight_layout()
plt.savefig('knock59_tsne.png', dpi=120)
print("Saved: knock59_tsne.png")
