import pandas as pd
from collections import Counter

def convert_to_bow(file_path):
    """TSVファイルを読み込み、指定された形式の辞書のリストに変換する関数"""
    df = pd.read_csv(file_path, sep='\t')
    dataset = []
    
    for _, row in df.iterrows():
        text = row['sentence']
        label = str(row['label']) 
        
        tokens = text.split()
        feature = dict(Counter(tokens))
        
        item = {
            'text': text,
            'label': label,
            'feature': feature
        }
        dataset.append(item)
        
    return dataset

train_data = convert_to_bow('SST-2/train.tsv')
dev_data = convert_to_bow('SST-2/dev.tsv')

print("【学習データの最初の事例】")
print(train_data[0])