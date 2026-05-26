import spacy
from collections import Counter

nlp = spacy.load("ja_ginza")
nouns = [t.lemma_ for line in open("kokoro.txt", encoding="utf-8") for t in nlp(line.strip()) if t.pos_ in ("NOUN", "PROPN")]

print(Counter(nouns).most_common(20))