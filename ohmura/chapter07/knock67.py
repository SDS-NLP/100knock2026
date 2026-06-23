import pandas as pd
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

y_train_pred = clf.predict(X_train)
y_dev_pred = clf.predict(X_dev)

def print_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='1')
    recall = recall_score(y_true, y_pred, pos_label='1')
    f1 = f1_score(y_true, y_pred, pos_label='1')
    
    print(f"【{dataset_name}】")
    print(f"正解率 (Accuracy) : {acc:.4f}")
    print(f"適合率 (Precision): {precision:.4f}")
    print(f"再現率 (Recall)   : {recall:.4f}")
    print(f"F1スコア (F1)     : {f1:.4f}")
    print("-" * 30)

print_metrics(y_train, y_train_pred, "学習データ")
print_metrics(y_dev, y_dev_pred, "検証データ")