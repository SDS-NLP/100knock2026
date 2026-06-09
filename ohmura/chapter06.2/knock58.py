from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage, dendrogram
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

Z = linkage(country_vecs, method='ward')

plt.figure(figsize=(15, 6))
dendrogram(Z, labels=valid_countries, leaf_rotation=90, leaf_font_size=8)
plt.tight_layout()
plt.savefig('dendrogram.png')