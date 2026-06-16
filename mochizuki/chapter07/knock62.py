from datasets import load_dataset
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
import joblib

dataset = load_dataset('glue', 'sst2')

def to_bow(text):
    return dict(Counter(text.lower().split()))

train_X = [to_bow(ex['sentence']) for ex in dataset['train']]
train_y = [ex['label'] for ex in dataset['train']]

vec = DictVectorizer()
X_train = vec.fit_transform(train_X)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, train_y)

joblib.dump({'clf': clf, 'vec': vec}, 'model.pkl')
print(f'vocabulary size: {len(vec.vocabulary_)}')
print('model saved to model.pkl')
