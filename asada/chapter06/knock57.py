import gensim
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

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

print(df_with_vector)

countries = df_with_vector.get_column("Name")
vectors = np.array(df_with_vector.get_column("vector").to_list())
kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(vectors)
df_clusters = (
    (
        pl.DataFrame(
            {"country": country, "label": int(label)}
            for country, label in zip(countries, kmeans.labels_)
        )
        .group_by("label")
        .agg(pl.col("country"))
    )
    .sort("label")
    .select([pl.col("label"), pl.col("country").list.join(", ")])
)
print(df_clusters)
df_clusters.write_csv("clusters.csv")
