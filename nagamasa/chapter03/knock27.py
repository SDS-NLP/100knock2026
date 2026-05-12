from knock20 import get_article
import re

# 27. 内部リンクの除去

uk = get_article("イギリス")
block = re.search(r'{{基礎情報.+?\n}}', uk, re.DOTALL)
matches = re.findall(r'\|(.+?)\s*=(.+?)\n', block.group(0))

def clean(value):
    # 強調マークアップの除去
    value = re.sub(r"'{2,5}", "", value)
    # 内部リンクの除去
    value = re.sub(r'\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]', r'\1', value)
    return value

info = {field.strip(): clean(value) for field, value in matches}
print(info)

# re.sub(パターン, 置換文字列, 対象文字列)
# 置換文字列に\1を使うとグループ1の中身に置換できる
# [[リンク先|表示テキスト]] → 表示テキスト
# [[リンク先]] → リンク先

# (?:...)は非キャプチャグループ
# ()と違い中身を記録しない。グループとして扱うだけ
# [^|\]]+は|と]以外の文字が一文字以上

# info.items()で辞書のキーと値のペアをイテレートできる
# for field, value in info だとキーだけ返る
# for field, value in info.items() が正しい

# clean()関数に処理をまとめることで
# 28以降で処理を追加しやすくなる