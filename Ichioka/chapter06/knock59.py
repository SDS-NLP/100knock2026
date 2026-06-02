from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"
questions_path = "tmp/questions-words.txt"

# モデル読み込み
model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

# 国名を含むセクション
target_sections = {
    "capital-common-countries",
    "capital-world",
}

countries = set()
current_section = None

# questions-words.txt から国名を抽出
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

        # 形式:
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

country_vectors = np.array(country_vectors)

print(f"使用する国名数: {len(country_names)}")

# t-SNEで2次元に圧縮
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="pca",
    learning_rate="auto"
)

country_vectors_2d = tsne.fit_transform(country_vectors)

# 可視化
plt.figure(figsize=(14, 10))

plt.scatter(
    country_vectors_2d[:, 0],
    country_vectors_2d[:, 1]
)

for i, country in enumerate(country_names):
    plt.annotate(
        country,
        xy=(country_vectors_2d[i, 0], country_vectors_2d[i, 1]),
        fontsize=8
    )

plt.title("t-SNE Visualization of Country Word Vectors")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.tight_layout()
plt.show()