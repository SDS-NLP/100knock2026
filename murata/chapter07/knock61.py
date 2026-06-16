import pandas as pd
from collections import Counter

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

print(train_list[0])