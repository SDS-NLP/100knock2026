from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
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

Z = linkage(vectors, method='ward')

plt.figure(figsize=(16, 8))
dendrogram(Z, labels=countries, leaf_rotation=90, leaf_font_size=9)
plt.title("Ward Hierarchical Clustering of Countries")
plt.xlabel("Country")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("q58_dendrogram.png", dpi=150)
plt.show()