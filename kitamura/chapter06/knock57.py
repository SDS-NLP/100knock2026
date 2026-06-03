import numpy as np
import re
from sklearn.cluster import KMeans
import gensim.downloader as api


print("Word2Vecモデルを読み込んでいます...")
model = api.load("word2vec-google-news-300")

countries = []
with open("countries.txt", "r", encoding="utf-8") as f:
    for line in f:
        countries.append(line)


country_vectors = []
valid_countries = []

for country in countries:
    if country in model:
        country_vectors.append(model[country])
        valid_countries.append(country)


X = np.array(country_vectors)

print("k-meansクラスタリングを実行中")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X)

clusters = {i: [] for i in range(5)}
for country, label in zip(valid_countries, kmeans.labels_):
    clusters[label].append(country)

for i in range(5):
    print(f"\n■ クラスタ {i} ({len(clusters[i])}カ国):")
    print(", ".join(clusters[i]))