import pandas as pd
from scipy.stats import spearmanr
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

df = pd.read_csv('wordsim353/combined.tab', sep='\t')

human_scores = []
model_scores = []

for _, row in df.iterrows():
    w1, w2 = row['Word 1'], row['Word 2']
    if w1 in model and w2 in model:
        human_scores.append(row['Human (mean)'])
        model_scores.append(model.similarity(w1, w2))

corr, _ = spearmanr(human_scores, model_scores)
print(corr)