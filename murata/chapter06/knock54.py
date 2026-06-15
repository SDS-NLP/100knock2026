from gensim.models import KeyedVectors
import numpy as np
from tqdm import tqdm
model = KeyedVectors.load_word2vec_format(f'C:/study/NLP100knock/100knock2026/GoogleNews-vectors-negative300.bin.gz', binary=True)


results = []
with open("100knock2026\questions-words.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith(":"):
            category = line[2:]
            continue
        words = line.split()
        pred, sim = model.most_similar(
            positive = [words[1], words[2]],
            negative = [words[0]],
            topn = 1
        )[0]
        results.append((category, words, pred, sim))
        
with open("q54_results.txt", "w", encoding="utf-8") as f:
    for category, words, pred, sim in results:
        f.write(f"{category} {' '.join(words)} {pred} {sim}\n")