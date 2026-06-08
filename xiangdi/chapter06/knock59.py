from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"
analogy_path = "questions-words.txt"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

country_sections = {
    "capital-common-countries",
    "capital-world",
}

countries = set()
section = None

with open(analogy_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line == "":
            continue

        if line.startswith(":"):
            section = line[2:]
            continue

        if section not in country_sections:
            continue

        word1, word2, word3, word4 = line.split()

        countries.add(word2)
        countries.add(word4)

country_names = []
country_vectors = []

for country in sorted(countries):
    if country in model:
        country_names.append(country)
        country_vectors.append(model[country])

country_vectors = np.array(country_vectors)

tsne = TSNE(
    n_components=2,
    random_state=111,
    perplexity=30,
    init="random",
    learning_rate="auto"
)

country_vectors_2d = tsne.fit_transform(country_vectors)

plt.figure(figsize=(14, 10))

x = country_vectors_2d[:, 0]
y = country_vectors_2d[:, 1]

plt.scatter(x, y)

for i, country in enumerate(country_names):
    plt.annotate(country, (x[i], y[i]), fontsize=8)

plt.title("t-SNE visualization of country word vectors")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")

plt.tight_layout()
plt.savefig("country_tsne.png", dpi=300)
plt.show()