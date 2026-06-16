import gensim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 59. t-SNEによる可視化
# 国名の単語ベクトルを t-SNE で2次元に落として散布図にする。

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

# t-SNE で2次元化（perplexity は標本数より小さい必要があるので clamp）
perplexity = min(30, len(countries) - 1)
tsne = TSNE(n_components=2, random_state=0, init="pca", perplexity=perplexity)
emb = tsne.fit_transform(X)

plt.figure(figsize=(14, 12))
plt.scatter(emb[:, 0], emb[:, 1])
for i, c in enumerate(countries):
    plt.annotate(c, (emb[i, 0], emb[i, 1]), fontsize=8)
plt.title("t-SNE of country word vectors")
plt.tight_layout()
plt.savefig("knock59_tsne.png", dpi=150)
print("t-SNE 散布図を knock59_tsne.png に保存しました")
plt.show()
