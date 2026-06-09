from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open('questions-words.txt', 'r') as f_in, open('ans54_full.txt', 'w') as f_out:
    for line in f_in:
        if line.startswith(':'):
            f_out.write(line)
            continue
        
        words = line.split()
        if len(words) == 4:
            try:
                pred, sim = model.most_similar(positive=[words[1], words[2]], negative=[words[0]], topn=1)[0]
                f_out.write(f"{line.strip()} {pred} {sim}\n")
            except KeyError:
                f_out.write(f"{line.strip()} N/A 0.0000\n")

print("done")