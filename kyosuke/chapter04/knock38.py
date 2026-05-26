import re
import math
import MeCab
from collections import Counter

with open('kokoro.txt', 'r', encoding='utf-8-sig') as f:
    content = f.read()

content_clean = re.sub(r'《[^》]*》', '', content)
content_clean = re.sub(r'[|]', '', content_clean)
content_clean = content_clean.replace('\r\n', '\n')

pattern = r'(?:^|\n)[一二三四五六七八九十百千万]+\n'
chapter = re.split(pattern, content_clean)
chapters = [ch.strip() for ch in chapter if ch.strip() != '']

tagger = MeCab.Tagger()

total_chapters = len(chapters)

total_words_list = []
for chapter in chapters:
    chapter_words_list = []
    node = tagger.parseToNode(chapter)
    while node:
        if node.surface != "":
            features = node.feature.split(',')
            if features[0] == '名詞':
                base = features[7] if len(features) > 7 and features[7] != '*' else node.surface
                chapter_words_list.append(base)
        node = node.next
    total_words_list.append(chapter_words_list)

def Cal_TF(chapter_words_list):
    tf = {}
    words_count = Counter(chapter_words_list)
    for word, freq in words_count.items():
        tf[word] = freq / len(chapter_words_list)
    return tf

def Cal_IDF(words_set):
    idf = {}
    for word in words_set:
        count = 0
        for i in range(len(total_words_list)):
            if word in total_words_list[i]:
                count += 1
        idf[word] = math.log(total_chapters / count)
    return idf
    
def Cal_TFIDF(chapter_words_list):
    word_set = set()
    for word in chapter_words_list:
        word_set.add(word)

    tf = Cal_TF(chapter_words_list)
    idf = Cal_IDF(word_set)
    tf_idf = {word: [tf[word] * idf[word], tf[word], idf[word]] for word in word_set}
    return tf_idf

select_chapter = int(input())
tf_idf = Cal_TFIDF(total_words_list[select_chapter - 1])
tf_idf = sorted(tf_idf.items(), key=lambda x: x[1][0], reverse=True)
for word, values in tf_idf[:20]:
    print(f"{word}\tTF-IDF: {values[0]:.6f}\tTF: {values[1]:.6f}\tIDF: {values[2]:.6f}")