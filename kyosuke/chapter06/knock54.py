import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'
word_file = 'capital-common-countries.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

with open(word_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line == ": capital-common-countries\n":
            continue
        if line == ": capital-world\n":
            break
        words = line.strip().split(" ")
        result = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
        (word, score) = result[0]
        print(f"{words[1]} - {words[0]} + {words[2]} = {word} (類似度: {score:.4f})")
