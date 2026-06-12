import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def convert_to_bow(file_path):
    df = pd.read_csv(file_path, sep='\t')
    dataset = []
    for _, row in df.iterrows():
        text = row['sentence']
        label = str(row['label'])
        feature = dict(Counter(text.split()))
        dataset.append({'text': text, 'label': label, 'feature': feature})
    return dataset

train_data = convert_to_bow('SST-2/train.tsv')

X_train_dict = [d['feature'] for d in train_data]
y_train = [d['label'] for d in train_data]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print("done!")