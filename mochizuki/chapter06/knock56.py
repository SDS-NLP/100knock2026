import zipfile
import io
import pandas as pd
from scipy.stats import spearmanr
import knock50

model = knock50.model

with zipfile.ZipFile('wordsim353.zip') as z:
    df = pd.read_csv(io.TextIOWrapper(z.open('combined.csv'), encoding='utf-8'))

cols = df.columns.tolist()  # ['Word 1', 'Word 2', 'Human (mean)']

human_scores, model_scores = [], []

for _, row in df.iterrows():
    w1, w2 = str(row[cols[0]]), str(row[cols[1]])
    try:
        model_scores.append(model.similarity(w1, w2))
        human_scores.append(float(row[cols[-1]]))
    except KeyError:
        pass

corr, pvalue = spearmanr(human_scores, model_scores)

if __name__ == '__main__':
    print(f'Spearman correlation: {corr:.4f}')
    print(f'p-value:              {pvalue:.4f}')
    print(f'Pairs evaluated:      {len(human_scores)}')
