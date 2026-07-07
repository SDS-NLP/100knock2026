import pandas as pd
from gensim.models import KeyedVectors
from scipy.stats import spearmanr


file = "chapter06/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(file, binary=True)

df = pd.read_csv("data/wordsim353/combined.csv")

human_scores = []
model_scores = []

count = 0

for _, row in df.iterrows():
    w1 = row["Word 1"]
    w2 = row["Word 2"]
    score = row["Human (mean)"]

    if w1 in model.key_to_index and w2 in model.key_to_index:
        human_scores.append(score)
        model_scores.append(model.similarity(w1, w2))
    count += 1
    if count % 10 == 0:
        print(f"{count}件処理済み")


rho, p = spearmanr(human_scores, model_scores)
print(f"評価対象数: {len(human_scores)}")
print(f"Spearman相関係数: {rho:.4f}")
print(f"p-value: {p:.4e}")

# 評価対象数: 353
# Spearman相関係数: 0.7000
# p-value: 2.8687e-53
