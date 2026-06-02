import gensim

# 54. アナロジーデータでの実験
# 単語アナロジー評価データの `: capital-common-countries` セクションについて、
# vec(2列目) - vec(1列目) + vec(3列目) の最近傍語と類似度を求め、各事例と一緒に記録する。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
DATA_PATH = "questions-words.txt"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# 評価データを全行読み込む
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]

# capital-common-countries セクションの各事例で vec(col2)-vec(col1)+vec(col3) の最近傍語を求める
# （flag でセクション内かを判定。most_similar に語を渡すので入力語は自動除外される）
acc = []
flag = False
for line in lines:
    if line == ": capital-common-countries":
        flag = True
        continue
    if line.startswith(":"):
        flag = False
        continue
    if flag:
        cols = line.split()
        for word, score in kv.most_similar(
            positive=[cols[1], cols[2]], negative=[cols[0]], topn=1
        ):
            acc.append(cols + [word, score])

# 結果を col1..col4 予測語 類似度 のタブ区切りで保存
OUT_PATH = "knock54_result.txt"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for row in acc:
        f.write("\t".join(map(str, row)) + "\n")
print(f"{len(acc)} 件を {OUT_PATH} に書き出しました")
