import pandas as pd

train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')

print(f'【学習データ: {train_df["label"].value_counts().to_dict()}】')
print(f'【検証データ: {dev_df["label"].value_counts().to_dict()}】')