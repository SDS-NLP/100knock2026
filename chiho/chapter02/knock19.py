import pandas as pd

df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
df_19 = df.sort_values(3, ascending=False)

print(df_19.head)