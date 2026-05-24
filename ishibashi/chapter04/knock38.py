import re
import math
from collections import Counter
import MeCab

def clean_text(text):
    """テキスト上のマークアップ等を除去する関数"""
    text = re.sub(r'《.*?》', '', text)
    text = re.sub(r'｜', '', text)

    return text

def get_sections(file):
    """指定したファイルを読み込み、漢数字の章ごとにテキストを分割する関数"""
    sections = []
    current_section_lines = []
    current_title = ''

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            line_str = line.strip()
            if re.match(r'^[一二三四五六七八九十]+$', line_str):
                if current_section_lines:
                    sections.append((current_title, "".join(current_section_lines)))
                current_title = line_str
                current_section_lines = []
            else:
                current_section_lines.append(line)
        if current_section_lines:
            sections.append((current_title, "".join(current_section_lines)))
    
    return sections

def calc_section_tfidf(file, target_section_title='一', top_n=20):
    """指定された章における名詞のTF-IDFスコアを計算する関数"""
    tagger = MeCab.Tagger()
    sections = get_sections(file)

    total_docs = len(sections)
    df_counter = Counter()
    target_tf = Counter()

    for title, text in sections:
        cleaned_text = clean_text(text)
        current_section_nouns = set()

        parsed_lines = tagger.parse(cleaned_text).split('\n')
        for line in parsed_lines:
            if line == 'EOS' or line == '':
                continue
        
            parts = line.split('\t')
            pos = parts[4].split('-')[0]
            if pos == '名詞':
                noun = parts[0].strip()
                if noun:
                    current_section_nouns.add(noun)
                    if title == target_section_title:
                        target_tf[noun] += 1

        for noun in current_section_nouns:
            df_counter[noun] += 1

    tfidf_results = []
    for word, tf in target_tf.items():
        df = df_counter[word]
        idf = math.log(total_docs / df) + 1.0
        tfidf = tf * idf
        tfidf_results.append((word, tf, idf, tfidf))

    tfidf_results.sort(key=lambda x: x[3], reverse=True)
    return tfidf_results[:top_n]

if __name__ == "__main__":
    file = './chapter04/kokoro.txt'
    target_section = "一"
    
    print(f"{'単語':<10} | {'TF':<5} | {'IDF':<6} | {'TF-IDF':<6}")
    print("-" * 45)
    
    for word, tf, idf, tfidf in calc_section_tfidf(file, target_section, 20):
        print(f"{word:<10} | {tf:<5} | {idf:<6.4f} | {tfidf:<6.4f}")