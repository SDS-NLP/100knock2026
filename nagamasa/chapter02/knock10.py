with open ("popular-names.txt", "r") as f:
    lines = f.readlines()
    print(len(lines))


with open("popular-names.txt") as f:
    print(sum(1 for _ in f))


# ファイル読み込みまとめ
# ファイルはディスクに存在し、メモリより10万倍遅い
# Pythonはバッファリングで8KB単位でまとめて読み、ディスクアクセス回数を減らす
#
# readlines()はファイル全体をメモリに載せてリストにする
# ジェネレータ(sum(1 for _ in f))は一行処理したら捨てるのでメモリ使用量が常に一行分
# 処理ステップ数は同じ。差はメモリ使用量だけ
# 巨大ファイルではジェネレータが破綻しない。小さいファイルでは差なし
# 習慣的にジェネレータを使う癖をつけるべき
#
# wc -l popular-names.txt