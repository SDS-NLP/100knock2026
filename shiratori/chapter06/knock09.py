from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

file = "data/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

analogy_file = "data/questions-words.txt"


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

# ベクトル取得
X = np.array([model[country] for country in countries])

# t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=30)

X_tsne = tsne.fit_transform(X)

# 描画
plt.figure(figsize=(12, 10))

for i, country in enumerate(countries):
    x = X_tsne[i, 0]
    y = X_tsne[i, 1]

    plt.scatter(x, y)
    plt.annotate(country, (x, y), fontsize=8)

plt.title("t-SNE Visualization of Country Vectors")
plt.tight_layout()
plt.savefig("outputs/chapter06/t-SNE.png", dpi=300, bbox_inches="tight")
plt.show()
