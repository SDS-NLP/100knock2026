from datasets import load_dataset
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

dataset = load_dataset('glue', 'sst2')
data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

def to_bow(text):
    return dict(Counter(text.lower().split()))

def evaluate(split):
    X = vec.transform([to_bow(ex['sentence']) for ex in dataset[split]])
    y = [ex['label'] for ex in dataset[split]]
    pred = clf.predict(X)
    print(f'{split}:')
    print(f'  accuracy:  {accuracy_score(y, pred):.4f}')
    print(f'  precision: {precision_score(y, pred):.4f}')
    print(f'  recall:    {recall_score(y, pred):.4f}')
    print(f'  f1:        {f1_score(y, pred):.4f}')

evaluate('train')
evaluate('validation')
