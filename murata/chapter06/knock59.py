from sklearn.manifold import TSNE
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

# 57のcountries, vectorsをそのまま使う
tsne = TSNE(n_components=2, random_state=42, perplexity=15, init='pca')
emb = tsne.fit_transform(vectors)

# 57のk-meansラベルで色分けすると綺麗
plt.figure(figsize=(14, 10))
plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='tab10', s=60)
for i, name in enumerate(countries):
    plt.annotate(name, (emb[i, 0], emb[i, 1]), fontsize=8, alpha=0.8)

plt.title("t-SNE Visualization of Country Vectors")
plt.tight_layout()
plt.savefig("q59_tsne.png", dpi=150)
plt.show()