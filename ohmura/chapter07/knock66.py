import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')

X_train_dict = [dict(Counter(str(row['sentence']).split())) for _, row in train_df.iterrows()]
y_train = [str(row['label']) for _, row in train_df.iterrows()]

X_dev_dict = [dict(Counter(str(row['sentence']).split())) for _, row in dev_df.iterrows()]
y_dev = [str(row['label']) for _, row in dev_df.iterrows()]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)
X_dev = vec.transform(X_dev_dict)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_dev)

cm = confusion_matrix(y_dev, y_pred, labels=['0', '1'])

print(f"               [予測: ネガ(0)]  [予測: ポジ(1)]")
print(f"[正解: ネガ(0)]      {cm[0][0]:<8}         {cm[0][1]}")
print(f"[正解: ポジ(1)]      {cm[1][0]:<8}         {cm[1][1]}")