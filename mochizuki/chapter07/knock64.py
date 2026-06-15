from datasets import load_dataset
from collections import Counter
import joblib

dataset = load_dataset('glue', 'sst2')
data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

def to_bow(text):
    return dict(Counter(text.lower().split()))

ex = dataset['validation'][0]
text = ex['sentence']

X = vec.transform([to_bow(text)])
proba = clf.predict_proba(X)[0]

print(f'text:              {text}')
for label, p in zip(clf.classes_, proba):
    name = 'positive' if label == 1 else 'negative'
    print(f'P({name}|text): {p:.4f}')
