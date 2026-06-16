import os
import urllib.request
import knock50

QUESTIONS_FILE = 'questions-words.txt'
URL = 'http://download.tensorflow.org/data/questions-words.txt'

if not os.path.exists(QUESTIONS_FILE):
    urllib.request.urlretrieve(URL, QUESTIONS_FILE)

model = knock50.model


def get_sections():
    sections = {}
    current = None
    with open(QUESTIONS_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith(':'):
                current = line[2:]
                sections[current] = []
            elif current and line:
                parts = line.split()
                if len(parts) == 4:
                    sections[current].append(tuple(parts))
    return sections


def run_analogy(words_list):
    results = []
    for w1, w2, w3, gold in words_list:
        try:
            pred, sim = model.most_similar(positive=[w2, w3], negative=[w1], topn=1)[0]
            results.append((w1, w2, w3, gold, pred, sim))
        except KeyError:
            pass
    return results


if __name__ == '__main__':
    sections = get_sections()
    results = run_analogy(sections.get('capital-world', []))
    for r in results:
        print('\t'.join(str(x) for x in r))
