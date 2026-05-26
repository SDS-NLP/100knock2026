import re
import knock25

def clean(text):
    # <ref>...</ref> と <ref ... /> を除去
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    text = re.sub(r'<references\s*/>', '', text)
    # [[ファイル:...]] / [[File:...]] を除去
    text = re.sub(r'\[\[(?:ファイル|File):[^\]]*\]\]', '', text)
    # {{lang|xx|text}} → text
    text = re.sub(r'\{\{lang\|[^|]+\|([^}]+)\}\}', r'\1', text)
    # {{仮リンク|表示|...}} → 表示
    text = re.sub(r'\{\{仮リンク\|([^|]+)\|[^}]+\}\}', r'\1', text)
    # {{0}} → 除去（日付整形用スペーサー）
    text = re.sub(r'\{\{0\}\}', '', text)
    # 残りのテンプレート {{...}} を除去
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # 強調マークアップ除去
    text = re.sub(r"'{2,5}", '', text)
    # 内部リンク [[link|display]] → display, [[link]] → link
    text = re.sub(r'\[\[(?:[^|\]]+\|)?([^\]]+)\]\]', r'\1', text)
    # 外部リンク [URL display] → display, [URL] → 除去
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # HTMLタグ除去
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

for k, v in knock25.info.items():
    print(f'{k}: {clean(v)}')
