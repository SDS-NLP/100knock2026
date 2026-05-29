from knock50 import load_model
from knock52 import most_similar


def iter_section(path, target_section):
    in_section = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                in_section = (line == target_section)
                continue

            if in_section and line:
                yield line

def evaluate_analogies(model, path, section):
    results = []
    for line in iter_section(path, section):
        w1, w2, w3, w4 = line.split()
        word, score = most_similar(model, positive=[w2, w3], negative=[w1], topn=1)[0]
        results.append((w1, w2, w3, w4, word, score))

    return results

if __name__ == '__main__':
    model = load_model()
    results = evaluate_analogies(
        model,
        'data/questions-words.txt', ': capital-common-countries'
    )
    for w1, w2, w3, w4, word, score in results:
        print(f'vec({w2}) - vec({w1}) + vec({w3}) ≒ {word} ({score:.4f}) [answer: {w4}]')
