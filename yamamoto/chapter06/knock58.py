#国名に関する単語ベクトルに対し、Ward法による階層型クラスタリングを実行せよ。さらに、クラスタリング結果をデンドログラムとして可視化せよ。

import knock57
from scipy.cluster.hierarchy import linkage, dendrogram #linkage:階層クラスタリング, dendrogram:樹形図の描画
import matplotlib.pyplot as plt

ward = linkage(knock57.country_vectors, method = "ward", metric = "euclidean") #ward法:クラスタ内の分散を最小化するように結合

plt.figure()

dendrogram(ward, labels = knock57.countries) #ラベルは国名を指定(countriesとcountry_vectorsの順序は対応している)

plt.show()