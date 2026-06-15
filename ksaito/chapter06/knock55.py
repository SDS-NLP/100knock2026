from knock50 import load_model


SEMANTIC_SECTIONS = {
    ': capital-common-countries',
    ': capital-world',
    ': currency',
    ': city-in-state',
    ': family',
}


def iter_analogy_results(path):
    section = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                section = line
                continue

            if line:
                yield section, line


def analogy_accuracy(results):
    correct = sum(1 for result in results if result[3] == result[4])
    return correct, len(results), correct / len(results)


def evaluate_by_type(model, path='data/questions-words.txt'):
    results = {
        'semantic': [],
        'syntactic': [],
    }
    for section, line in iter_analogy_results(path):
        w1, w2, w3, w4 = line.split()
        word, score = model.most_similar(
            positive=[w2, w3],
            negative=[w1],
            topn=1,
        )[0]
        key = 'semantic' if section in SEMANTIC_SECTIONS else 'syntactic'
        results[key].append((w1, w2, w3, w4, word, score))

    return results


if __name__ == '__main__':
    model = load_model()
    results = evaluate_by_type(model)

    for key in ['semantic', 'syntactic']:
        correct, total, accuracy = analogy_accuracy(results[key])
        print(f'{key}: {correct}/{total} = {accuracy:.4f}')
