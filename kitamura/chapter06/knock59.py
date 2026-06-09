import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

print("tsne圧縮")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(16,12))
x = X_tsne[:, 0]
y = X_tsne[:, 1]

plt.scatter(x,y, alpha=0.5)
for i, country in enumerate(valid_countries):
    plt.annotate(country, (x[i], y[i]), fontsize=9)

plt.title("t-SNE Visualization of Country Word Vectors")
plt.savefig("tsne_countries.png")