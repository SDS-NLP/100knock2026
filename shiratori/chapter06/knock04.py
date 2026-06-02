from gensim.models import KeyedVectors

file = "chapter06/GoogleNews-vectors-negative300.bin.gz"
output = "chapter06/output04.txt"

model = KeyedVectors.load_word2vec_format(file, binary=True)

analogy = "chapter06/questions-words.txt"

with open(analogy, encoding="utf-8") as f:
    with open(output, "w", encoding="utf-8") as out:

        capital_section = False

        for line in f:
            line = line.strip()

            if line.startswith(":"):
                capital_section = line == ": capital-common-countries"
                continue

            if not capital_section:
                continue

            w1, w2, w3 = line.split()[:3]

            result = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)

            word, sim = result[0]

            out.write(f"{w1}\t{w2}\t{w3}\t{word}\t{sim:.6f}\n")
