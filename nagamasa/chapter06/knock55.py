import gensim

# 55. アナロジータスクでの正解率
# 54 と同じベクトル演算を全セクションに適用し、
# 意味的アナロジー（gram以外）と文法的アナロジー（gram1-9）の正解率を測定する。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
DATA_PATH = "questions-words.txt"

# word2vec バイナリ形式を読み込む（50・54と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# 評価データを全行読み込む
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f]

# gram系セクション=文法的、それ以外=意味的に振り分け、予測語が正解(cols[3])と一致した割合を集計する
count_sem = 0
count_sem_all = 0
count_gram = 0
count_gram_all = 0
flag = False
for line in lines:
    if line.startswith(": gram"):
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
            if word == cols[3]:
                count_gram += 1
            count_gram_all += 1
    else:
        cols = line.split()
        for word, score in kv.most_similar(
            positive=[cols[1], cols[2]], negative=[cols[0]], topn=1
        ):
            if word == cols[3]:
                count_sem += 1
            count_sem_all += 1

print(f"意味的正解率：{count_sem / count_sem_all}")
print(f"文法正解率：{count_gram / count_gram_all}")
