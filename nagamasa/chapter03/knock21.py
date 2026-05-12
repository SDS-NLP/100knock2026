from knock20 import get_article
import re

# 21. カテゴリ名を含む行を抽出

uk = get_article("イギリス")

# 方法1: startswith()
for line in uk.splitlines():
    if line.startswith("[[Category:"):
        print(line)

# 方法2: 正規表現
categories = [line for line in uk.splitlines() if re.search(r'^\[\[Category:', line)]
for line in categories:
    print(line)

# splitlines()は文字列を改行で分割してリストにする
# "abc\ndef".splitlines() -> ["abc", "def"]

# startswith()は文字列が指定した文字列で始まるかTrue/Falseを返す
# == Trueは不要。startswith()は既にboolを返す

# 正規表現のポイント
# ^ は行頭にマッチする
# \[ は[をエスケープ。[]は正規表現で特殊文字(文字クラス)なのでそのまま書けない
# re.search()は文字列中にパターンがあればマッチオブジェクトを返す
# マッチしなければNoneを返す。ifの条件としてそのまま使える