from gensim.models import KeyedVectors
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

file = "chapter06/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

analogy_file = "chapter06/questions-words.txt"


def extract_countries(filepath):
    countries = set()

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(":"):
                continue

            words = line.split()

            for word in words:
                if word in model.key_to_index:
                    countries.add(word)

    return sorted(countries)


countries = extract_countries(analogy_file)

X = np.array([model[country] for country in countries])

# Ward法
linkage_matrix = linkage(X, method="ward")

# デンドログラム
plt.figure(figsize=(20, 8))

dendrogram(linkage_matrix, labels=countries, leaf_rotation=90, leaf_font_size=8)

plt.title("Ward Hierarchical Clustering of Countries")
plt.xlabel("Country")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("chapter06/countries_dendrogram.png", dpi=300, bbox_inches="tight")

plt.show()
