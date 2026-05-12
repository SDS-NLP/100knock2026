from knock20 import get_article
import re

# 28. MediaWikiマークアップの除去

uk = get_article("イギリス")
block = re.search(r'{{基礎情報.+?\n}}', uk, re.DOTALL)
matches = re.findall(r'\|(.+?)\s*=\s*([\s\S]+?)(?=\n\||\n}})', block.group(0))

def clean(value):
    # 強調マークアップの除去
    value = re.sub(r"'{2,5}", "", value)
    # langテンプレートの処理（内部リンク除去より先に行う）
    value = re.sub(r'{{lang\|.+?\|(.+?)}}', r'\1', value)
    # 内部リンクの除去
    value = re.sub(r'\[\[(?:[^|\]]+\|)?([^|\]]+)\]\]', r'\1', value)
    # 仮リンクの除去
    value = re.sub(r'{{仮リンク\|(.+?)\|.+?}}', r'\1', value)
    # その他マークアップの除去
    value = re.sub(r'{{center\|.*?}}|\{\{0\}\}|{{en icon}}|<ref[^/][\s\S]*?</ref>|<ref[^>]*/>|<br />', "", value, flags=re.DOTALL)
    return value

info = {field.strip(): clean(value) for field, value in matches}
if __name__ == "__main__":
    print(info)

# フィールド抽出パターンの改善
# r'\|(.+?)\s*=(.+?)\n' → 値が\nで切れる問題があった
# r'\|(.+?)\s*=\s*([\s\S]+?)(?=\n\||\n}})' に変更
# (?=\n\||\n}})は先読みアサーション
# 次が\n|か\n}}になるまでを値として取得する
# これにより複数行にまたがる値も正しく取得できる

# 先読みアサション(?=...)はマッチに含まれない
# つまり\n|の\nと|は消費されないので次のフィールドの抽出に影響しない

# <ref>タグの問題
# <ref name="...">...</ref>は値の中に>が含まれる
# [^>]*では>が含まれる値に対応できない
# <ref[^/][\s\S]*?</ref>で</ref>だけを終端として扱う
# [^/]で自己閉じタグ<ref .../>と区別する

# 処理の順番が重要
# {{lang}}を先に処理してテキストを取り出してから内部リンクを除去する
# 順番が逆だと[[リンク|{{lang|en|テキスト}}]]の処理が正しくできない