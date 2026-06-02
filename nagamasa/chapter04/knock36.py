import re
from collections import Counter

import MeCab

# 36. 単語の出現頻度（コーパス: kokoro.txt）
# 仕様: 語彙素[7]でカウント / 品詞大分類[0]が記号系(補助記号・空白)は除外 / 上位20語を表示

# --- ファイル読み込み（utf-8で全文を1つの文字列に） ---
with open("kokoro.txt", encoding="utf-8") as f:
    text = f.read()

# --- 前処理: 青空文庫のルビ等を除去 ---
text = text.replace("﻿", "")     # BOM（残すと先頭 "一" が "﻿一" になる）
text = re.sub(r"《[^》]*》", "", text)  # ルビ読み《…》（[^》]で跨ぎ削りを防ぐ）
text = text.replace("｜", "")          # ルビ起点記号
# 章見出しの漢数字行は仕様どおり残す

# --- 形態素解析（unidic-lite を自動検出） ---
tagger = MeCab.Tagger()

# --- 語彙素を集計（記号は除外、列不足の語は表層形で代用） ---
counter = Counter()
node = tagger.parseToNode(text)
while node:
    if node.surface != "":            # BOS/EOS を飛ばす
        cols = node.feature.split(",")
        if cols[0] not in {"補助記号", "空白"}:
            counter[cols[7] if len(cols) > 7 else node.surface] += 1
    node = node.next

# --- 頻度上位20語 ---
for word, count in counter.most_common(20):
    print(f"{word}\t{count}")
