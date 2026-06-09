import gensim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
input_file = 'capital-common-countries.txt'
countries = set()
is_target = False

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            if line in [': capital-common-countries', ': capital-world']:
                is_target = True
            else:
                is_target = False
            continue
            
        if is_target:
            words = line.split()
            countries.add(words[1])
            countries.add(words[3])

countries = list(countries)

valid_countries = [c for c in countries if c in model]
vectors = [model[c] for c in valid_countries]
vectors_array = np.array(vectors)

print("t-SNEを計算中...（少し時間がかかります）")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
vectors_2d = tsne.fit_transform(vectors_array)

print("グラフを描画しています...")
plt.figure(figsize=(16, 12))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='blue', alpha=0.5)
for i, country in enumerate(valid_countries):
    plt.annotate(country, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)

plt.title('t-SNE Visualization of Country Vectors')
plt.tight_layout()
plt.show()