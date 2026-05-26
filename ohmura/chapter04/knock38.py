import spacy, math
from collections import Counter

nlp = spacy.load("ja_ginza")

docs = [[t.lemma_ for t in nlp(line.strip()) if t.pos_ in ("NOUN", "PROPN")] 
        for line in open("kokoro.txt", encoding="utf-8") if line.strip()]

N = len(docs)
tf = Counter(w for d in docs for w in d)
df = Counter(w for d in set(d) for d in docs) 

tfidf = sorted([(w, tf[w], math.log10(N/df[w]), tf[w]*math.log10(N/df[w])) for w in tf], key=lambda x: x[3], reverse=True)[:20]

for rank, (w, t, i, ti) in enumerate(tfidf, 1):
    print(f"{rank:2d}位: {w:<8} TF:{t:<4} IDF:{i:.3f} TF-IDF:{ti:.3f}")