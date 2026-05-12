import pandas as pd

df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
print(df.sample(n=len(df)))
