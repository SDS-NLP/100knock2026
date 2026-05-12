from knock20 import get_article
import re

# 26. 強調マークアップの除去

uk = get_article("イギリス")
block = re.search(r'{{基礎情報.+?\n}}', uk, re.DOTALL)
matches = re.findall(r'\|(.+?)\s*=(.+?)\n', block.group(0))
info = {field.strip(): re.sub(r"'{2,5}", "", value) for field, value in matches}
print(info)

# re.sub(パターン, 置換文字列, 対象文字列)
# パターンにマッチした部分を置換文字列で置き換える
# 除去したい場合は空文字列""に置換する

# MediaWikiの強調マークアップ
# ''テキスト''  → 弱い強調（イタリック）
# '''テキスト''' → 強調（ボールド）
# '''''テキスト''''' → 強い強調（ボールド+イタリック）

# '{2,5}で2個以上5個以下の'にマッチ
# 三種類まとめて一つのパターンで除去できる
# 長いパターンから処理しないと意図しない結果になる場合があるが
# {2,5}は最長マッチなので自動的に長い方から処理される

# 辞書内包表記で変換しながら辞書を作る
# field.strip()でフィールド名の空白も除去