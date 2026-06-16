from datasets import load_dataset
from collections import Counter
from sklearn.metrics import confusion_matrix
import joblib

dataset = load_dataset('glue', 'sst2')
data = joblib.load('model.pkl')
clf, vec = data['clf'], data['vec']

def to_bow(text):
    return dict(Counter(text.lower().split()))

dev_X = vec.transform([to_bow(ex['sentence']) for ex in dataset['validation']])
dev_y = [ex['label'] for ex in dataset['validation']]

pred_y = clf.predict(dev_X)
cm = confusion_matrix(dev_y, pred_y)

print('Confusion Matrix (rows=actual, cols=predicted):')
print(f'              pred_neg  pred_pos')
print(f'actual_neg      {cm[0][0]:5d}     {cm[0][1]:5d}')
print(f'actual_pos      {cm[1][0]:5d}     {cm[1][1]:5d}')
