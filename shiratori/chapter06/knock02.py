from gensim.models import KeyedVectors

file = "chapter06/GoogleNews-vectors-negative300.bin.gz"
output = "chapter06/output02.txt"

model = KeyedVectors.load_word2vec_format(file, binary=True)

result = model.most_similar("United_States", topn=10)

print(result)

with open(output, "w") as f:
    for word, sim in result:
        f.write(f"{word}\t{sim:.6f}\n")
