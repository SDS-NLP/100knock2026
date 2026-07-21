from gensim.models import KeyedVectors

file = "data/GoogleNews-vectors-negative300.bin.gz"
output = "outputs/chapter06/output03.txt"

model = KeyedVectors.load_word2vec_format(file, binary=True)

result = model.most_similar(positive=["Spain", "Athens"], negative=["Madrid"], topn=10)

print(result)

with open(output, "w") as f:
    for word, sim in result:
        f.write(f"{word}\t{sim:.6f}\n")
