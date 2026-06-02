import math
import re
from collections import Counter

import MeCab

# 38. TF・IDF（コーパス: kokoro.txt / 1文書 = 1章, 全110章）
# 仕様:
#   A 章分割: 漢数字だけの行を見出しとして分割 / 見出しは本文に含めない
#   B TF    : 章内の相対頻度 f/Σf
#   C IDF   : 平滑化 log(N/(1+df))  （N=110, df=その語が出現する章数）
#   D 対象語: 普通名詞だけ（cols[0]=="名詞" かつ cols[1]=="普通名詞"）/ キーは語彙素[7]・列<8は表層形
#   E 出力  : 関数(章index) -> その章の TF・IDF 上位20語と TF, IDF, TF・IDF

# --- ファイル読み込み（utf-8で全文を1つの文字列に） ---
with open("kokoro.txt", encoding="utf-8") as f:
    text = f.read()

# --- 前処理: 青空文庫のルビ等を除去（36/37と同じ） ---
text = text.replace("﻿", "")  # BOM
text = re.sub(r"《[^》]*》", "", text)  # ルビ読み《…》
text = text.replace("｜", "")  # ルビ起点記号

# --- 形態素解析器（unidic-lite を自動検出） ---
tagger = MeCab.Tagger()

# --- 章に分割（漢数字だけの行で区切る。先頭の空要素は捨てる） ---
chapters = re.split(r"^[一二三四五六七八九十百]+$", text, flags=re.MULTILINE)
chapters = chapters[1:]  # 先頭の空要素を捨てる（index0 のみ空 → 110章）
N = len(chapters)  # 110（IDFの分母）

# --- 章ごとに普通名詞Counterを保存しつつ df（出現章数）を貯める ---
chap_counters = []  # 章index -> Counter(語: 回数)
df = Counter()  # 語 -> 出現章数
for chapter in chapters:
    counter = Counter()
    node = tagger.parseToNode(chapter)
    while node:
        if node.surface != "":  # BOS/EOS を飛ばす
            cols = node.feature.split(",")
            if cols[0] == "名詞" and cols[1] == "普通名詞":
                counter[cols[7] if len(cols) > 7 else node.surface] += 1
        node = node.next
    chap_counters.append(counter)
    df.update(set(counter))  # その章のユニーク語に +1（同章で何回出ても df は+1）

# --- IDF表（平滑化 log(N/(1+df))） ---
idf = {word: math.log(N / (1 + df[word])) for word in df}


# --- その章の TF・IDF 上位 topn 語を表示 ---
def show_tfidf(idx, topn=20):
    counter = chap_counters[idx]
    total = sum(counter.values())  # Σf（章内の普通名詞総数）
    tf = {word: count / total for word, count in counter.items()}
    tfidf = {word: tf[word] * idf[word] for word in tf}
    sorted_tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_tfidf[:topn]:
        print(f"{word}\t{tf[word]:.6f}\t{idf[word]:.4f}\t{score:.6f}")


show_tfidf(0)  # 例: 1章目(上一)を表示
