from datasets import load_dataset
from collections import Counter
import joblib

dataset = load_dataset('glue', 'sst2')
data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

def to_bow(text):
    return dict(Counter(text.lower().split()))

ex = dataset['validation'][0]
text, label = ex['sentence'], ex['label']

X = vec.transform([to_bow(text)])
pred = clf.predict(X)[0]

print(f'text:       {text}')
print(f'pred:       {pred}')
print(f'label:      {label}')
print(f'correct:    {pred == label}')
