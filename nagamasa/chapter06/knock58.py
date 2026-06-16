import gensim
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# 58. Ward法によるクラスタリング
# 国名の単語ベクトルを Ward 法で階層型クラスタリングし、デンドログラムを描く。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
DATA_PATH = "questions-words.txt"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)


def load_countries(path):
    # capital 系セクションの各行 2列目・4列目が国名。重複を除き語彙にあるものだけ返す（57と同じ）。
    countries = set()
    flag = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(": capital-common-countries") or line.startswith(": capital-world"):
                flag = True
                continue
            if line.startswith(":"):
                flag = False
                continue
            if flag:
                cols = line.split()
                countries.add(cols[1])
                countries.add(cols[3])
    return [c for c in sorted(countries) if c in kv]


countries = load_countries(DATA_PATH)
X = np.array([kv[c] for c in countries])

# Ward法（併合による分散増加が最小のペアを結合）で連結し、デンドログラムを保存する
Z = linkage(X, method="ward")
plt.figure(figsize=(14, 7))
dendrogram(Z, labels=countries, leaf_font_size=8)
plt.title("Ward clustering of country word vectors")
plt.tight_layout()
plt.savefig("knock58_dendrogram.png", dpi=150)
print("デンドログラムを knock58_dendrogram.png に保存しました")
plt.show()
