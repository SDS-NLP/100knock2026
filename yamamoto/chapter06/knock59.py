#ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ。

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import knock57

tsne = TSNE(n_components = 2, random_state = 0, perplexity = 30) #t-SNEで次元圧縮するインスタンスを作成

vectors_tsne = tsne.fit_transform(knock57.country_vectors) #国名の単語ベクトルを2次元に圧縮

plt.figure()

for vector, country, color in zip(vectors_tsne, knock57.countries, knock57.labels):
    
    plt.text(vector[0], vector[1], country, color = "C{}".format(color)) #点ではなく文字で描画

plt.xlim([-12, 15])
plt.ylim([-10, 15])

plt.savefig("tsne_vectors.png")

plt.show()