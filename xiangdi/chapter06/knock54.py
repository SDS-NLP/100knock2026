from gensim.models import KeyedVectors
import os
import urllib.request

analogy_path = "questions-words.txt"
url = "https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt"

if not os.path.exists(analogy_path):
    urllib.request.urlretrieve(url, analogy_path)

model_path = "/Users/caitlyn/Downloads/GoogleNews-vectors-negative300.bin.gz"

output_path = "capital-common-countries-result.txt"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

in_target_section = False

with open(analogy_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", encoding="utf-8") as out:

    for line in f:
        line = line.strip()

        if line.startswith(":"):
            if line == ": capital-common-countries":
                in_target_section = True
            else:
                in_target_section = False
            continue

        if not in_target_section:
            continue

        word1, word2, word3, word4 = line.split()

        predicted_word, similarity = model.most_similar(
                positive=[word2, word3],
                negative=[word1],
                topn=1
            )[0]

        print(
                word1,
                word2,
                word3,
                word4,
                predicted_word,
                similarity,
                sep="\t",
                file=out,
            )