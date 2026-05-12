from knock20 import get_article
import re

# 25. テンプレートの抽出

uk = get_article("イギリス")
block = re.search(r'{{基礎情報.+?\n}}', uk, re.DOTALL)
matches = re.findall(r'\|(.+?)\s*=(.+?)\n', block.group(0))
info = dict(matches)
print(info)

# re.search()は最初の一つだけマッチオブジェクトを返す
# .group(0)でマッチした文字列全体を取り出す
# .group(1)は最初のグループ、.group(2)は二番目のグループ

# re.DOTALLフラグで.が改行にもマッチするようになる
# デフォルトでは.は改行を無視する
# 複数行にまたがるパターンを扱う場合は必須

# 二段階に分けた理由
# 一段階でやると}}を超えてマッチしてしまう
# まずブロックを抽出してからフィールドを抽出する

# dict(タプルのリスト)で辞書に変換できる
# [(key1, val1), (key2, val2)] -> {key1: val1, key2: val2}
# forループで一つずつ追加するより簡潔