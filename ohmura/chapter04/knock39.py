import spacy
import matplotlib.pyplot as plt
from collections import Counter

nlp = spacy.load("ja_ginza")

print("テキストを解析中...")

words = [t.lemma_ for line in open("kokoro.txt", encoding="utf-8") 
         for t in nlp(line.strip()) if not t.is_punct and not t.is_space]

freqs = [count for word, count in Counter(words).most_common()]

plt.figure(figsize=(8, 6))
plt.loglog(range(1, len(freqs) + 1), freqs, marker=".", linestyle="None", alpha=0.5)

plt.xlabel("Rank (Log scale)")
plt.ylabel("Frequency (Log scale)")
plt.title("Zipf's law (kokoro.txt)")
plt.grid(True, which="both", ls="--")
plt.show()