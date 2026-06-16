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
dev_data = convert_to_bow('SST-2/dev.tsv')

X_train_dict = [d['feature'] for d in train_data]
y_train = [d['label'] for d in train_data]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

X_dev_dict = [d['feature'] for d in dev_data]
X_dev = vec.transform(X_dev_dict)

first_dev_instance = X_dev[0]
text = dev_data[0]['text']

probabilities = clf.predict_proba(first_dev_instance)[0]
classes = clf.classes_

print("【検証データの先頭事例に対する条件付き確率】")
print(f"テキスト: {text}")
print("-" * 40)

for cls, prob in zip(classes, probabilities):
    label_name = "ポジティブ" if cls == '1' else "ネガティブ"
    print(f"ラベル {cls} ({label_name}) の確率: {prob:.4f} ({prob*100:.2f}%)")