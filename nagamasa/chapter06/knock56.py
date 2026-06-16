import gensim
from scipy.stats import spearmanr

# 56. WordSimilarity-353での評価
# WordSim-353 の人間による類似度判定と、単語ベクトルのコサイン類似度の
# 順位の一致度を、スピアマン順位相関係数で測る。

MODEL_PATH = "GoogleNews-vectors-negative300.bin"
WS353_PATH = "wordsim353/combined.csv"

# word2vec バイナリ形式を読み込む（50と同じ）
kv = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)

# 評価データを読み込む（1行目はヘッダ "Word 1,Word 2,Human (mean)" なので [1:] で除く）
with open(WS353_PATH, encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f][1:]

# 各ペアの人間スコアとモデルのコサイン類似度を、同じ順序で2本のリストに貯める
# （OOV語を含むペアは similarity が KeyError → どちらにも入れずスキップし対応を保つ）
human = []
model = []
for line in lines:
    w1, w2, score = line.split(",")
    try:
        sim = kv.similarity(w1, w2)
    except KeyError:
        continue
    human.append(float(score))
    model.append(sim)

# スピアマン相関係数（spearmanr は同順位を平均順位で処理する）
result = spearmanr(human, model)
print(f"評価ペア数: {len(human)}（OOVで除外: {len(lines) - len(human)}）")
print(f"スピアマン相関係数: {result.correlation}")


"""
評価ペア数: 353（OOVで除外: 0）
スピアマン相関係数: 0.7000166486272194
"""
