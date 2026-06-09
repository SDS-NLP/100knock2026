import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

def run_tsne_visualization():
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
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    
    embedded_vectors = tsne.fit_transform(np.array(country_vectors))

    plt.figure(figsize=(16, 12))

    plt.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1], c='blue', alpha=0.5)

    for i, country in enumerate(valid_countries):
        plt.annotate(country, (embedded_vectors[i, 0], embedded_vectors[i, 1]), fontsize=9)

    plt.title("t-SNE Visualization of Country Word Vectors (300D -> 2D)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    output_img = "./chapter06/knock59_tsne.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"結果を '{output_img}' として保存しました！画像を開いて確認してください。")

if __name__ == "__main__":
    run_tsne_visualization()