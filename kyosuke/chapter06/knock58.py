import gensim
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

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

print("階層的クラスタリングを実行中...")
linkage_matrix = linkage(vectors, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=valid_countries, leaf_rotation=90)
plt.title("dendorogram")
plt.xlabel("countries")
plt.ylabel("distance")
plt.tight_layout()
plt.show()