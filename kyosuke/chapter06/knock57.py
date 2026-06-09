import gensim
from sklearn.cluster import KMeans

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

print("k-meansクラスタリングを実行中...")
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(vectors)

print("\n【 クラスタリング結果 (k=5) 】\n")
for i in range(5):
    cluster_countries = [valid_countries[j] for j, label in enumerate(kmeans.labels_) if label == i]
    
    print(f"{i} ({len(cluster_countries)}カ国):")
    print(", ".join(cluster_countries))