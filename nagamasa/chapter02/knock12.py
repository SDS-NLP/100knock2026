from collections import deque

def tail(N):
    with open("popular-names.txt") as f:
        d = deque(maxlen=N)
        for line in f:
            d.append(line)
    return d

for line in tail(10):
    print(line, end="")

# 確認コマンド: tail -n 10 popular-names.txt

# 末尾N行取得の原理
# 末尾はファイルを全部読み終えるまでわからない
# そのため原理的にファイル全体を読む必要がある。これは回避できない

# dequeのmaxlen=Nを使うと常にメモリにN行しか残らない
# 全行読みながら古いものを自動で捨てる。メモリ使用量はO(N)で一定

# dequeはリストと違い先頭・末尾への追加削除がO(1)
# リストの先頭操作はO(n)なので大きなファイルでは差が出る

# tailコマンドはseek()で末尾から逆読みする実装が多い
# f.seek(0, 2)でファイル末尾にカーソルを移動し
# f.seek(-1, 1)で1バイトずつ戻りながら改行をN個探す
# 通常ファイルではこの方法がより効率的
# ただしマルチバイト文字(日本語など)が含まれる場合は文字の途中で切れる危険がある
# ストリーム入力ではランダムアクセスできないためdequeと同じ戦略になる