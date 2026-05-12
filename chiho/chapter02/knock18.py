import pandas as pd
from collections import Counter

df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
names = df[0]
counter = Counter(names)

for name, count in counter.most_common():
    print(name, count)
