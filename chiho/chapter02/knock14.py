#14. 1列目を出力
#ファイルの先頭10行に対して、各行の1列目だけを抜き出して表示せよ。確認にはcutコマンドなどを用いよ。

import pandas as pd

N = 10
df = pd.read_csv("chiho/chapter02/popular-names.txt", sep="\t", header=None)
print(df.head(N)[0])

# head -n 10 chiho/chapter02/popular-names.txt | cut -f1
