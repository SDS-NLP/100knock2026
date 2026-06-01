from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

with open('questions-words.txt', 'r') as f_in, open('ans54.txt', 'w') as f_out:
    is_target = False
    for line in f_in:
        if line.startswith(':'):
            is_target = (line.strip() == ': capital-common-countries')
            continue
        
        if is_target:
            w1, w2, w3, w4 = line.split()
            pred, sim = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)[0]
            
            f_out.write(f"{line.strip()} {pred} {sim}\n")
            
            print(f"問題: {w2} - {w1} + {w3}  =>  予測: {pred} (類似度: {sim:.4f})")