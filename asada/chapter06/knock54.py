from pathlib import Path

import gensim

text = Path("questions-words.txt").read_text().split("\n")
start, end = 0, 0
for i, line in enumerate(text):
    if line.strip() == ": capital-common-countries":
        start = i + 1
        break
for i, line in enumerate(text[start:], start):
    if line.strip().startswith(":"):
        end = i
        break
text = text[start:end]

model = gensim.models.KeyedVectors.load_word2vec_format(
    "./GoogleNews-vectors-negative300.bin", binary=True
)
result = {}
for line in text:
    words = line.split()
    similar_word = model.most_similar(
        positive=[words[1], words[2]], negative=[words[0]], topn=1
    )
    result[line] = similar_word
print(result)
