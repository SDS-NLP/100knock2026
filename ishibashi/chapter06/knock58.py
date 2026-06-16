import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from gensim.models import KeyedVectors

def run_ward_clustering():
    model_path = './chapter06/GoogleNews-vectors-negative300.bin.gz'
    countries_file = './chapter06/countries.txt'
    
    model = KeyedVectors.load_word2vec_format(model_path, binary=True, limit=500000)

    countries = []
    with open(countries_file, 'r', encoding='utf-8') as f:
        for line in f:
            countries.append(line.strip())

    valid_countries = []
    country_vectors = []
    for country in countries:
        if country in model:
            valid_countries.append(country)
            country_vectors.append(model[country])
    
    Z = linkage(country_vectors, method='ward')

    plt.figure(figsize=(16, 10))
    dendrogram(Z, labels=valid_countries, leaf_font_size=8, leaf_rotation=90)
    plt.title("Dendrogram of Countries (Ward's Method)")
    plt.xlabel("Countries")
    plt.ylabel("Distance")
    
    output_img = "./chapter06/knock58_dendrogram.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"結果を '{output_img}' として保存しました！画像を開いて確認してください。")

if __name__ == "__main__":
    run_ward_clustering()