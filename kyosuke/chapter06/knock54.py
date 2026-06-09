import gensim
file_path = 'GoogleNews-vectors-negative300.bin.gz'
word_file = 'capital-common-countries.txt'
output_file = 'knock54_output.txt'
model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

with open(word_file, 'r', encoding='utf-8') as f,\
     open(output_file, 'w', encoding='utf-8') as out_f:
    for line in f:
        line = line.strip()
        if line.startswith(":"):
            print(line)
            out_f.write(line + "\n")
            continue
        words = line.split()
        result = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)
        (word, score) = result[0]
        print(f"{words[1]} - {words[0]} + {words[2]} = {word} ({score:.4f})")
        out_f.write(f"{line} {word} {score:.4f}\n")