import gensim
import numpy as np
from sklearn.cluster import KMeans

# 57. k-meansクラスタリング
# 国名の単語ベクトルを抽出し、k-means（k=5）でクラスタリングする。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
DATA_PATH = "questions-words.txt"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)


def load_countries(path):
    # questions-words.txt の capital 系セクションの各行 "city1 country1 city2 country2" の
    # 2列目・4列目が国名。重複を除き、語彙にある国名だけを返す。
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
X = np.array([kv[c] for c in countries])  # (国数, 300) のベクトル行列

# k-means（k=5）でクラスタリングし、各クラスタの国名を出力する
km = KMeans(n_clusters=5, random_state=0, n_init=10)
labels = km.fit_predict(X)
for cid in range(5):
    members = [countries[i] for i in range(len(countries)) if labels[i] == cid]
    print(f"--- cluster {cid} ({len(members)}) ---")
    print(", ".join(members))


"""
--- cluster 0 (19) ---
Afghanistan, Bahrain, Bangladesh, Bhutan, Egypt, Indonesia, Iran, Iraq, Jordan, Lebanon, Libya, Morocco, Nepal, Oman, Pakistan, Qatar, Syria, Thailand, Tunisia
--- cluster 1 (26) ---
Australia, Bahamas, Belize, Canada, Chile, China, Cuba, Dominica, Ecuador, Fiji, Greenland, Guyana, Honduras, Jamaica, Japan, Laos, Nicaragua, Peru, Philippines, Samoa, Suriname, Taiwan, Tuvalu, Uruguay, Venezuela, Vietnam
--- cluster 2 (11) ---
Armenia, Azerbaijan, Belarus, Kazakhstan, Kyrgyzstan, Moldova, Russia, Tajikistan, Turkmenistan, Ukraine, Uzbekistan
--- cluster 3 (34) ---
Albania, Austria, Belgium, Bulgaria, Croatia, Cyprus, Denmark, England, Estonia, Finland, France, Georgia, Germany, Greece, Hungary, Ireland, Italy, Latvia, Liechtenstein, Lithuania, Macedonia, Malta, Montenegro, Norway, Poland, Portugal, Romania, Serbia, Slovakia, Slovenia, Spain, Sweden, Switzerland, Turkey
--- cluster 4 (26) ---
Algeria, Angola, Botswana, Burundi, Eritrea, Gabon, Gambia, Ghana, Guinea, Kenya, Liberia, Madagascar, Malawi, Mali, Mauritania, Mozambique, Namibia, Niger, Nigeria, Rwanda, Senegal, Somalia, Sudan, Uganda, Zambia, Zimbabwe
"""
