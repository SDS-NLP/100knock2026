

# def get_head(N):
#     with open ("popular-names.txt") as f:
#         lines = f.readlines()
#         for i in range(N):
#             print(lines[i])

# get_head(10)


from itertools import islice

# 11. 先頭からN行を出力
# ファイルの先頭N行を表示する

def get(N):
    with open("popular-names.txt") as f:
        for line in islice(f, N):
            print(line, end="")

get(10)

# 確認コマンド: head -n 10 popular-names.txt

# 設計の選択肢
# 1. readlines()[:N] -> 全行メモリに載せてからスライス。O(ファイル全体)
# 2. カウンタ変数 -> 全行読みながらカウント。コードが冗長
# 3. enumerate -> カウンタを自動管理。ただし全行読む
# 4. islice -> N行読んだ時点で止まる。O(N)で最もメモリ効率が良い

# isliceはイテレータに対して動くジェネレータ
# リストのスライスと違い、N行目以降はディスクから読み込まれすらしない
# 巨大ファイルでN行だけ欲しい場合に最適

# end=""はprint()がデフォルトで付加する改行と
# ファイルの行末\nの二重改行を防ぐ