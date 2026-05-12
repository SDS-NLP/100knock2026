import pandas as pd

N = 10
df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
print(df.head(N)[0])