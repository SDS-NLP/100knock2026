import pandas as pd
from collections import Counter
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score

df_train = pd.read_csv("SST-2/SST-2/train.tsv", sep='\t')
df_dev = pd.read_csv("SST-2/SST-2/dev.tsv", sep='\t')

def convert2dic(df):
    l = df['sentence']
    dic_list = []
    for i in range(len(l)):
        text = df.iloc[i]['sentence']
        d = dict()
        d['text'] = str(text)
        d['label'] = str(df.iloc[i]['label'])
        text_l = text.split()
        d['feature'] = dict(Counter(text_l))
        dic_list.append(d)
    return dic_list

train_list = convert2dic(df_train)
dev_list = convert2dic(df_dev)

logit_model = sklearn.linear_model.LogisticRegression(max_iter = 1000)



X_train_dict = [d['feature'] for d in train_list]
y_train = [int(d['label']) for d in train_list]

X_dev_dict = [d['feature'] for d in dev_list]
y_dev = [int(d['label']) for d in dev_list]

vec = DictVectorizer()
X_train = vec.fit_transform(X_train_dict)
X_dev = vec.transform(X_dev_dict)

logit_model.fit(X_train, y_train)

result = logit_model.predict_proba(X_dev)
print(result[0])