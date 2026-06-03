import csv
import time

from groq import Groq

# 42. 多肢選択問題の正解率
# JMMLU（日本語版 MMLU）の「日本史」150問を LLM に解かせ、正解率を求める。
# データ: japanese_history.csv（6列・ヘッダ無し: 問題, 選択肢A〜D, 正解記号）。
#   選択肢にカンマを含む行があるので csv モジュールで読む（split(",") 不可）。
# 大量に問い合わせるので Groq（無料枠 14,400回/日, 30 RPM）を使う。

client = Groq()  # GROQ_API_KEY を自動で読む
MODEL = "llama-3.3-70b-versatile"
JMMLU_CSV = "japanese_history.csv"

# --- CSV を読み込む（各行を6要素のリストに） ---
with open(JMMLU_CSV, encoding="utf-8", newline="") as f:
    rows = list(csv.reader(f))  # rows[i] = [問題文, 選択肢A, B, C, D, 正解記号]

# --- 1問ずつ解かせ、正解数を数える ---
correct = 0
for row in rows:
    contents = f"問題: {row[0]}\n選択肢:\nA: {row[1]}\nB: {row[2]}\nC: {row[3]}\nD: {row[4]}\n出力形式: A/B/C/D のいずれかの記号のみを出力せよ"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": contents}],
        temperature=0,             # 測定なので毎回同じ答えに固定
        max_completion_tokens=16,  # 出力を短く縛る（暴走防止＋TPM節約）
    )
    answer = resp.choices[0].message.content
    pred = answer.strip().upper()  # 応答を正規化（前後の空白を除き大文字化）して記号と比較
    if pred == row[5]:
        correct += 1
    time.sleep(2.1)  # 30 RPM を超えないように

print(f"正解数: {correct}/{len(rows)}")
print(f"正解率: {correct / len(rows) * 100:.1f}%")

# --- 実行結果 ---
"""
正解数: 89/150
正解率: 59.3%
"""
