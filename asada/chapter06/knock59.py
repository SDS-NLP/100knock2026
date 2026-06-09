import gensim
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)
df = pl.read_csv("countries.csv", encoding="utf-8-lossy").select("Name")


def extract_vector(row: dict) -> list[float]:
    word = row["Name"].replace(" ", "_")
    return model[word].tolist() if word in model else [0]


df_with_vector = df.with_columns(
    pl.struct(["Name"])
    .map_elements(extract_vector, return_dtype=pl.List(pl.Float64))
    .alias("vector")
).filter(pl.col("vector") != [0])

countries = df_with_vector.get_column("Name")
vectors = np.array(df_with_vector.get_column("vector").to_list())
tsne = TSNE(random_state=0).fit_transform(vectors)
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit_predict(vectors)

fig, ax = plt.subplots(figsize=(20, 20))
cmap = plt.get_cmap("Dark2")
for i in range(tsne.shape[0]):
    cval = cmap(kmeans[i] / 4)
    ax.scatter(tsne[i][0], tsne[i][1], marker=".", color=cval)
    ax.annotate(countries[i], xy=(tsne[i][0], tsne[i][1]), color=cval)
plt.savefig("tsne.png")
