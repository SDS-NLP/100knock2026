import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import gensim.downloader as api

print("Word2Vecモデルを読み込んでいます...")
model = api.load("word2vec-google-news-300")

countries = []
with open("countries.txt", "r", encoding="utf-8") as f:
    for line in f:
        countries.append(line.strip())


country_vectors = []
valid_countries = []

for country in countries:
    if country in model:
        country_vectors.append(model[country])
        valid_countries.append(country)


X = np.array(country_vectors)

Z = linkage(X, method="ward")
plt.figure(figsize=(20,5))
dendrogram(Z, labels=valid_countries)
plt.ylabel("Distance(Ward)")
plt.savefig("dendrogram58.png")