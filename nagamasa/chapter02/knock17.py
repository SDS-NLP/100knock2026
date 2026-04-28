import csv

# 17. 1列目の文字列の異なり

with open("popular-names.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    unique = {row[0] for row in reader}

print(len(unique))

# 確認コマンド: cut -f 1 popular-names.txt | sort | uniq | wc -l

# 集合内包表記 {} でsetを作る
# リスト内包表記 [] との違いは順序を持たず重複を自動排除する点
# 内部はハッシュテーブル。ハッシュ関数でキーを整数に変換してメモリに格納
# 同じキーは同じハッシュ値になるので重複チェックがO(1)でできる
# リスト全体を走査する必要がないため効率的

# uniqは隣接する重複行しか除去できない
# そのため必ずsortとセットで使う
# sort | uniq の計算量はO(n log n)
# PythonのsetはハッシュでO(1)だが
# 巨大ファイルではsortの外部ソートアルゴリズムが有利な場面もある

# 副作用のための内包表記は書くな
# [unique.add(x) for x in ...]は不要なリストをメモリに確保する無駄
# 副作用が目的ならforループを使え