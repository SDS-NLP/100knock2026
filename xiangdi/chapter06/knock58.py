from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

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

linkage_result = linkage(country_vectors, method="ward")

plt.figure(figsize=(16, 8))

dendrogram(
    linkage_result,
    labels=country_names,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical clustering of country vectors using Ward method")
plt.xlabel("Country")
plt.ylabel("Distance")

plt.tight_layout()
plt.savefig("country_dendrogram.png", dpi=300)
plt.show()