from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

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

print(f"使用する国名数: {len(country_names)}")

# Ward法による階層型クラスタリング
linkage_matrix = linkage(
    country_vectors,
    method="ward"
)

# デンドログラムの描画
plt.figure(figsize=(18, 8))

dendrogram(
    linkage_matrix,
    labels=country_names,
    leaf_rotation=90,
    leaf_font_size=8
)

plt.title("Hierarchical Clustering of Country Vectors by Ward Method")
plt.xlabel("Country")
plt.ylabel("Distance")

plt.tight_layout()
plt.show()