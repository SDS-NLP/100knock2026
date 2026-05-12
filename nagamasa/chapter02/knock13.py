from itertools import islice

def replace_tab(N):
    with open("popular-names.txt") as f:
        lines = [line.replace("\t", " ") for line in islice(f, N)]
    return lines

for line in replace_tab(10):
    print(line, end="")

# 確認コマンド
# sed 's/\t/ /g' popular-names.txt | head -n 10
# head -n 10 popular-names.txt | tr '\t' ' '
# head -n 10 popular-names.txt | expand -t 1

# | はパイプ。左の出力を右の入力に渡す
# UNIXの設計思想「一つのことをうまくやる小さなツールを組み合わせる」の体現

# sed: s/置換前/置換後/g が基本構文。g(global)で行内全マッチを置換
# tr: 文字を一対一で変換する専用コマンド。単純な文字置換に向いている
# expand: タブをスペースに変換する専用コマンド。-t 1でタブ幅を1スペースに指定

# 内包表記はPythonインタープリタが最適化している
# 通常のforループより高速で可読性も高い
# [式 for 変数 in イテラブル] でループしながらリストを一行で作る