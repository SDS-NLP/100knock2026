from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import csv

model = KeyedVectors.load_word2vec_format(
    'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz',
    binary=True
)

human_scores = []
model_scores = []

with open("C:/study/NLP100knock/100knock2026/wordsim353/combined.csv", encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        w1, w2, human = row[0], row[1], float(row[2])
        sim = model.similarity(w1, w2)
        human_scores.append(human)
        model_scores.append(sim)
corr, p = spearmanr(human_scores, model_scores)
print(corr, p)