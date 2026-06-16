import pandas as pd

train_df = pd.read_csv('SST-2/train.tsv', sep='\t')
print("【train.tsv】")
print(train_df['label'].value_counts())

dev_df = pd.read_csv('SST-2/dev.tsv', sep='\t')
print("\n【dev.tsv】")
print(dev_df['label'].value_counts())