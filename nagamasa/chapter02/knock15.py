import numpy as np

def split_file(num_splits):
    with open("popular-names.txt") as f:
        lines = f.readlines()
    
    chunks = np.array_split(lines, num_splits)
    
    for i, chunk in enumerate(chunks):
        with open(f'split/popular-names_{i}.txt', "w") as output_file:
            output_file.writelines(chunk)

split_file(10)

# 確認コマンド: mkdir -p split_unix && split -l 278 popular-names.txt split_unix/popular-names_
# -l で1ファイルあたりの行数を指定する

# readlines()は全行をリストとしてメモリに載せる
# ファイルサイズが大きい場合はメモリを圧迫する
# 今回は全行が必要なのでreadlines()が適切。isliceの出番はない

# np.array_split(配列, N)でN分割する
# 余りを自動で均等分配するので自分で余りを計算する必要がない
# 例: 2780行を10分割 → 278行ずつ均等に分割される
# numpyの内部はC言語実装なので数値計算は高速
# ただし文字列処理では数値計算の恩恵は受けられない

# enumerate(iterable)はインデックスと要素のペアを返すジェネレータ
# カウンタを自分で管理する必要がなくなる

# writelines()はリストをそのまま書き込める
# write()は文字列のみ受け取る

# 切り上げ除算のイディオム: -(-a // b)
# 負にして切り捨て、符号を戻すと切り上げになる
# math.ceil(a/b)でも同じだが浮動小数点を経由する
# -(-a // b)は整数演算のみで精度の問題がない

# yieldはreturnと違い関数を一時停止して再開できる
# ジェネレータ関数を作る際に使う
# メモリ効率を上げたい場合はreadlines()をやめて
# ファイルを直接イテレートしながらyieldで一チャンクずつ返す設計が理想