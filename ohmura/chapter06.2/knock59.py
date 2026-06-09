import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

countries = set()
with open('questions-words.txt', 'r') as f:
    is_target = False
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            is_target = line in [': capital-common-countries', ': capital-world']
            continue
        
        if is_target:
            words = line.split()
            if len(words) == 4:
                countries.add(words[1])
                countries.add(words[3])

countries = list(countries)

country_vecs = []
valid_countries = []
for country in countries:
    if country in model:
        country_vecs.append(model[country])
        valid_countries.append(country)

country_vecs = np.array(country_vecs)

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(country_vecs)

plt.figure(figsize=(15, 15))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
for i, label in enumerate(valid_countries):
    plt.annotate(label, (X_tsne[i, 0], X_tsne[i, 1]))

plt.tight_layout()
plt.savefig('tsne.png')