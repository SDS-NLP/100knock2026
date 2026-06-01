import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # 画面表示せずファイル保存するためのバックエンド
import matplotlib.pyplot as plt
import MeCab

# 39. Zipfの法則（コーパス: kokoro.txt）
# 仕様:
#   - 対象語: 36と同じ（記号除く語彙素・全語）
#   - 頻度を降順に並べ、横軸=順位 / 縦軸=頻度 を両対数でプロット
#   - plt.loglog で両軸logにして zipf.png に保存

# --- ファイル読み込み（utf-8で全文を1つの文字列に） ---
with open("kokoro.txt", encoding="utf-8") as f:
    text = f.read()

# --- 前処理: 青空文庫のルビ等を除去（36と同じ） ---
text = text.replace("﻿", "")  # BOM
text = re.sub(r"《[^》]*》", "", text)  # ルビ読み《…》
text = text.replace("｜", "")  # ルビ起点記号

# --- 形態素解析器（unidic-lite を自動検出） ---
tagger = MeCab.Tagger()

# --- 語彙素を集計（記号は除外、列不足の語は表層形で代用。36と同じ） ---
counter = Counter()
node = tagger.parseToNode(text)
while node:
    if node.surface != "":  # BOS/EOS を飛ばす
        cols = node.feature.split(",")
        if cols[0] not in {"補助記号", "空白"}:
            counter[cols[7] if len(cols) > 7 else node.surface] += 1
    node = node.next

# --- 頻度を降順に並べ、順位を振る ---
freqs = [
    count for word, count in sorted(counter.items(), key=lambda x: x[1], reverse=True)
]
ranks = list(range(1, len(freqs) + 1))

# --- 両対数プロットして保存（雑用: matplotlib足場） ---
plt.loglog(ranks, freqs)
plt.xlabel("rank")
plt.ylabel("frequency")
plt.title("Zipf's law (kokoro.txt)")
plt.savefig("zipf.png")
