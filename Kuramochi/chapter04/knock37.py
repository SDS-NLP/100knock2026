import json
import re
from collections import Counter
from pathlib import Path
from janome.tokenizer import Tokenizer

INPUT_PATH = Path(__file__).resolve().parents[1] / "chapter03" / "jawiki-country.json"

# マークアップを除去する関数（ブラックボックス）
def remove_jawiki_markup(text: str) -> str:
    text = text.replace("'''", "").replace("''", "")

    # 内部リンク [[A|B]] または [[A]]
    def _replace_link(match):
        inner = match.group(1)
        if '|' in inner:
            display = inner.split('|')[-1]
        else:
            display = inner
        if display.startswith(('File:', 'ファイル:', 'Category:', 'カテゴリ:')):
            return ''
        return display

    text = re.sub(r'\[\[([^\]]+)\]\]', _replace_link, text)

    # 外部リンク [http://... text]
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)

    # セクション見出し == heading ==
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)

    # テンプレート等の {{...}} をできる限り除去（ネスト対応）
    while re.search(r'\{\{[^{}]*\}\}', text):
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)

    # HTMLタグやrefを除去
    text = re.sub(r'<ref[^>]*>.*?<\/ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)

    # 行末の注釈や脚注、強調などの残りを削除
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'&nbsp;|&lt;|&gt;|&quot;|&amp;', ' ', text)

    # 余分な空白をまとめる
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# マークアップを除去したテキストを表示する関数
def get_jawiki_contents(path: Path):
    all_text = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            cleaned = remove_jawiki_markup(article.get('text', ''))
            all_text.append(cleaned)

    return '\n'.join(all_text)

# 頻出名詞の上位20件を取得する関数
def get_top20_noun_words(text):

    tokenizer = Tokenizer()
    words = []
    
    # 1. 形態素解析をして単語（基本形）をリストに格納
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]  # 品詞の大分類を取得
        
        # 句読点やカッコなどの「記号」はカウントから除外する
        if pos == '名詞':
            words.append(token.base_form)
            
    # 2. カウントと並び替え
    # Counterにリストを渡すと、要素の出現回数を自動でカウントしてくれます
    word_counter = Counter(words)
    
    # most_common(n) で、出現頻度が高い順に上位n件を取得
    top20_words = word_counter.most_common(20)
    
    return top20_words


if __name__ == '__main__':
    cleaned_text = get_jawiki_contents(INPUT_PATH)
    top20_words  = get_top20_noun_words(cleaned_text)
    for word, count in top20_words:
        print(f"{word}: {count}")