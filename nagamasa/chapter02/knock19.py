import csv

# 19. 3列目の数値の降順に各行を並び替える

with open("popular-names.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    sorted_list = sorted(reader, key=lambda x: int(x[2]), reverse=True)

for line in sorted_list:
    print(line)

# 確認コマンド: sort -t $'\t' -k 3 -rn popular-names.txt
# -t $'\t' で区切り文字をタブに指定。$'\t'はシェルでタブを表す記法
# -k 3 で3列目をキーにソート
# -rn で数値として降順ソート。-nがないと文字列比較になりバグる

# sorted(iterable, key=lambda x: x[2], reverse=True)
# keyに関数を渡してソートの基準を指定する
# lambdaは無名関数。lambda x: int(x[2])は「3列目を整数として返す」関数
# reverse=Trueで降順になる

# 3列目は文字列として読み込まれる
# 数値としてソートするにはint()で変換する必要がある
# int()なしだと文字列比較になり正しくソートされない

# PythonのsortedはTimsortを使う
# マージソートとInsertionSortの組み合わせ
# 部分的にソート済みな入力に対してO(n)に近い性能を発揮する
# 最悪計算量はO(n log n)

# タプルのアンパック: for name, sex, amount, birth in sorted_list
# 4要素のリストを4変数に分解して受け取る