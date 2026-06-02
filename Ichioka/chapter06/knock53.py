from gensim.models import KeyedVectors

model_path = "tmp/GoogleNews-vectors-negative300.bin.gz"

model = KeyedVectors.load_word2vec_format(
    model_path,
    binary=True
)

# Spain - Madrid + Athens
# positive: 足すベクトル
# negative: 引くベクトル
results = model.most_similar(
    positive=["Spain", "Athens"],
    negative=["Madrid"],
    topn=10
)

for word, similarity in results:
    print(f"{word}\t{similarity:.6f}")