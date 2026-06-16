import csv
import os
import zipfile
from urllib.request import urlretrieve

from scipy.stats import spearmanr

from knock50 import load_model


URL = 'https://www.gabrilovich.com/resources/data/wordsim353/wordsim353.zip'
ZIP_PATH = 'data/wordsim353.zip'
DATA_DIR = 'data/wordsim353'
COMBINED_CSV = os.path.join(DATA_DIR, 'combined.csv')


def download_wordsim353():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(ZIP_PATH):
        urlretrieve(URL, ZIP_PATH)

    if not os.path.exists(COMBINED_CSV):
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(DATA_DIR)
        for dirpath, _, filenames in os.walk(DATA_DIR):
            if 'combined.csv' in filenames:
                os.replace(
                    os.path.join(dirpath, 'combined.csv'),
                    COMBINED_CSV,
                )
                break


def iter_wordsim353(path=COMBINED_CSV):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row['Word 1'], row['Word 2'], float(row['Human (mean)'])


def calc_similarities(model):
    human_scores = []
    model_scores = []
    for word1, word2, human_score in iter_wordsim353():
        if word1 in model and word2 in model:
            human_scores.append(human_score)
            model_scores.append(model.similarity(word1, word2))

    return human_scores, model_scores


if __name__ == '__main__':
    download_wordsim353()
    model = load_model()
    human_scores, model_scores = calc_similarities(model)
    correlation, pvalue = spearmanr(human_scores, model_scores)
    print(f'used pairs: {len(human_scores)}')
    print(f'Spearman correlation: {correlation:.4f}')
    print(f'p-value: {pvalue:.4g}')
