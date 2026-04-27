from itertools import islice
import csv

# 14. 1列目を出力

# 方法1: split()を使う
def dipp(N):
    with open("popular-names.txt") as f:
        lines = [line.split() for line in islice(f, N)]
    return lines

for line in dipp(10):
    print(line[0])

# 方法2: csvモジュールを使う
with open("popular-names.txt") as f:
    reader = csv.reader(f, delimiter="\t")
    for line in islice(reader, 10):
        print(line[0])

# 確認コマンド: cut -f 1 popular-names.txt | head -n 10
# cut -f 1 でタブ区切りの1列目を抜き出す。-fはfieldの略

# split()とcsvモジュールの違い
# split("\t")は文字列をタブで単純分割する
# 引数なしのsplit()はタブ・スペース・改行をまとめて区切り文字として扱い行末\nも除去する
# csvモジュールはdelimiterで区切り文字を指定する汎用パーサー
# 値の中にタブや改行が含まれる複雑なケースでもクォートやエスケープを正しく処理する
# 拡張子はファイル名の一部に過ぎない。.txtでもdelimiter="\t"でTSVとして処理できる

# 内包表記はPythonが専用バイトコードLIST_APPENDを使うためforループより10~30%速い
# appendの属性参照コストが発生しないことが理由