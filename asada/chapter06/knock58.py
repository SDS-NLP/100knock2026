import gensim
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import dendrogram, linkage

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
linkage_result = linkage(vectors, method="ward")

plt.figure(figsize=(15, 10))
dendrogram(linkage_result, labels=countries)
plt.savefig("ward.png")
