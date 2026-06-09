import gensim
from scipy.stats import spearmanr

model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
input_file = "combined.csv"
human_scores = []
ai_scores = []

with open(input_file, "r",encoding="utf-8") as f:
    next(f)
    for line in f:
        word1, word2, human_score = line.strip().split(",")
        human_score = float(human_score)
        cosine_similarity = model.similarity(word1, word2)
        ai_scores.append(cosine_similarity)
        human_scores.append(human_score)

correlation, p_value = spearmanr(human_scores, ai_scores)
print(f"スピアマン相関係数: {correlation:.3f}")
print(f"p値: {p_value}")