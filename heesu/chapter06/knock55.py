from gensim.models import KeyedVectors

SEMANTIC_SECTIONS = {'capital-common-countries', 'capital-world', 'currency', 'city-in-state', 'family'}

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

results = []  # (section, w4_correct, w4_pred)
section = None

with open('questions-words.txt') as f:
    for line in f:
        line = line.strip()
        if line.startswith(':'):
            section = line[2:]
            continue
        if section is None:
            continue
        w1, w2, w3, w4 = line.split()
        pred_word, _ = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)[0]
        results.append((section, w4, pred_word))

def accuracy(rows):
    if not rows:
        return 0.0, 0, 0
    correct = sum(1 for _, truth, pred in rows if truth == pred)
    return correct / len(rows), correct, len(rows)

semantic = [(s, t, p) for s, t, p in results if s in SEMANTIC_SECTIONS]
syntactic = [(s, t, p) for s, t, p in results if s not in SEMANTIC_SECTIONS]

sem_acc, sem_c, sem_n = accuracy(semantic)
syn_acc, syn_c, syn_n = accuracy(syntactic)
tot_acc, tot_c, tot_n = accuracy(results)

print(f"Semantic Analogy  : {sem_acc:.4f} ({sem_c} / {sem_n})")
print(f"Syntactic Analogy : {syn_acc:.4f} ({syn_c} / {syn_n})")
print(f"Total             : {tot_acc:.4f} ({tot_c} / {tot_n})")
