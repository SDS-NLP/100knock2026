from gensim.models import KeyedVectors
from scipy.stats import spearmanr
import csv

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

csv_path = "/Users/caitlyn/Downloads/wordsim353/combined.csv"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

human_scores = []
model_scores = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)

    header = next(reader)

    for row in reader:
        word1 = row[0]
        word2 = row[1]
        human_score = float(row[2])

        try:
            similarity = model.similarity(word1, word2)

        except KeyError:
            continue

        human_scores.append(human_score)
        model_scores.append(similarity)


correlation, _ = spearmanr(human_scores, model_scores)

print("Spearman correlation:", correlation)

#　Spearman correlation: 0.7000166486272194