import pandas as pd

df_train = pd.read_csv("SST-2/SST-2/train.tsv", sep='\t')
df_dev = pd.read_csv("SST-2/SST-2/dev.tsv", sep='\t')

print(f"train-positive: {df_train['label'].sum()}, train-negative: {len(df_train['label']) - df_train['label'].sum()}, dev-positive: {df_dev['label'].sum()}, dev-negative: {len(df_dev['label']) - df_dev['label'].sum()}")

