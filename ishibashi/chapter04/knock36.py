import re
from collections import Counter
import MeCab

def clean_text(text):
    """テキスト上のマークアップ等を除去する関数"""
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'｜', '', text)

    return text

def get_wrod_frequency(file, top_n=20):
    """指定したファイルからすべての単語の出現頻度トップ20を返す関数"""
    tagger = MeCab.Tagger()
    word_counter = Counter()

    with open(file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        cleaned_text = clean_text(raw_text)

        parsed_lines = tagger.parse(cleaned_text).split('\n')
        for line in parsed_lines:
            if line == 'EOS' or line == '':
                continue

            parts = line.split('\t')
            word = parts[0].strip()
            if word:
                word_counter[word] += 1

    return word_counter.most_common(top_n)

if __name__ == "__main__":
    file = './chapter04/kokoro.txt'
    i = 1

    for word, count in get_wrod_frequency(file, 20):
        print(f"{i}位")
        print(f"{word}: {count}")
        i += 1