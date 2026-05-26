from collections import Counter
import spacy

nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
word_counter = Counter()

with open("kokoro.txt", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 1000:
            break
        if line.strip():
            for t in nlp(line):
                if t.pos_ != "PUNCT":
                    word_counter[t.lemma_] += 1

for word, count in word_counter.most_common(20):
    print(f"{word}\t{count}")