from gensim.models import KeyedVectors


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open('questions-words.txt') as f:
    in_section = False
    for line in f:
        line = line.strip()
        if line == ': capital-common-countries':
            in_section = True
            continue
        if line.startswith(':'):
            if in_section:
                break
            continue
        if not in_section:
            continue
        w1, w2, w3, w4 = line.split()
        result = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)
        word, score = result[0]
        print(f'{w1}\t{w2}\t{w3}\t{w4}\t{word}\t{score:.4f}')
