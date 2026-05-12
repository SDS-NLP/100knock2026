file = "popular-names.txt"
N = 10

with open(file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i>= N:
            break

        cols = line.split("\t")
        print(cols[0])


# import pandas as pd

# N = 10
# df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
# print(df.head(N)[0])