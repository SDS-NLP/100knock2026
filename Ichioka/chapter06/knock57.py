from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"
questions_path = "tmp/questions-words.txt"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

target_sections = {
    "capital-common-countries",
    "capital-world",
}

countries = set()
current_section = None

with open(questions_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if line.startswith(":"):
            current_section = line[2:]
            continue

        if current_section not in target_sections:
            continue

        words = line.split()

        if len(words) != 4:
            continue

        # questions-words.txt の形式:
        # capital1 country1 capital2 country2
        capital1, country1, capital2, country2 = words

        countries.add(country1)
        countries.add(country2)

# モデルに存在する国名だけを使う
country_names = []
country_vectors = []

for country in sorted(countries):
    if country in model:
        country_names.append(country)
        country_vectors.append(model[country])
    else:
        print(f"モデルに存在しないためスキップ: {country}")

print(f"国名数: {len(country_names)}")

# k-means クラスタリング
k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init="auto"
)

labels = kmeans.fit_predict(country_vectors)

# 結果をクラスタごとに表示
clusters = {}

for country, label in zip(country_names, labels):
    clusters.setdefault(label, []).append(country)

for label in sorted(clusters):
    print(f"\nCluster {label}:")
    for country in clusters[label]:
        print(country)