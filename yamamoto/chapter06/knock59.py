#ベクトル空間上の国名に関する単語ベクトルをt-SNEで可視化せよ。

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import knock57

tsne = TSNE(n_components = 2, random_state = 0, perplexity = 30) #t-SNEで次元圧縮するインスタンスを作成

vectors_tsne = tsne.fit_transform(knock57.country_vectors) #国名の単語ベクトルを2次元に圧縮

plt.figure()
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], c = knock57.labels) #一応k-meansのラベルをつけて散布図を描画

plt.show()