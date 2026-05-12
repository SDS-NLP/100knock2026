import csv
from collections import Counter

# 18. 各行の1列目の文字列の出現頻度を求め、出現頻度の高い順に並べる

with open("popular-names.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    counter = Counter(row[0] for row in reader)

for name, count in counter.most_common():
    print(count, name)

# 確認コマンド: cut -f 1 popular-names.txt | sort | uniq -c | sort -rn
# uniq -cで出現回数を行頭に付加する
# sort -rnで数値として降順ソート
# -nがないと文字列比較になり10より9が大きいと判断される

# Counter.most_common()は出現頻度の高い順にソート済みの
# (要素, 出現回数)のタプルのリストを返す
# 引数にNを渡すと上位N件だけ返す

# タプルのアンパック: for name, count in counter.most_common()
# タプルを変数に分解して受け取る

# csv.readerの利点
# 行末の\nを自動除去するのでstrip()が不要
# クォートやエスケープを正しく処理する
# タブ区切りファイルを読む意図が明確になる

# 副作用のための内包表記は書くな
# printの戻り値はNone。内包表記で呼ぶとNoneのリストがメモリに確保される無駄