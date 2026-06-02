import llm
import random
import re
from collections import defaultdict

theme = "時間"
NUM_ROUNDS = 3

# Step 1: Generate 10 senryu
gen_prompt = f"""お題「{theme}」で川柳を10個作成してください。"""

print(f"お題：{theme}\n")
senryu_result = llm.chat(
    gen_prompt,
    system="あなたは川柳の名人です。ユーモアあふれる川柳を作ります。",
    temperature=0.9,
)
print("=== 生成された川柳 ===")
print(senryu_result)

# Parse senryu into a list of (original_index, text) pairs
lines = [l.strip() for l in senryu_result.strip().splitlines() if l.strip()]
# Group lines into senryu entries (each entry may span multiple lines)
entries = []
current = []
for line in lines:
    if re.match(r"^\d+[\.\．]", line) and current:
        entries.append("\n".join(current))
        current = [line]
    else:
        current.append(line)
if current:
    entries.append("\n".join(current))

# Step 2: Multi-round evaluation with shuffled order
scores = defaultdict(list)  # original_index -> list of scores

for round_num in range(1, NUM_ROUNDS + 1):
    order = list(range(len(entries)))
    random.shuffle(order)

    shuffled_list = "\n\n".join(
        f"{i+1}. {entries[orig_idx]}" for i, orig_idx in enumerate(order)
    )

    eval_prompt = f"""以下の川柳を1つずつ10点満点で評価してください。

川柳リスト：
{shuffled_list}

各川柳について必ず以下の形式で出力してください：
番号: <番号>
点数: <数字>/10
"""

    print(f"\n=== 評価ラウンド {round_num}（順序シャッフル済み） ===")
    raw = llm.chat(eval_prompt, temperature=0.3)
    print(raw)

    # Parse scores from this round
    for match in re.finditer(r"番号[:：]\s*(\d+).*?点数[:：]\s*(\d+)", raw, re.DOTALL):
        shuffled_pos = int(match.group(1)) - 1
        score = int(match.group(2))
        if 0 <= shuffled_pos < len(order):
            orig_idx = order[shuffled_pos]
            scores[orig_idx].append(score)

# Step 3: Print aggregated results
print("\n=== 総合評価（平均点） ===")
ranked = []
for i, entry in enumerate(entries):
    s = scores[i]
    avg = sum(s) / len(s) if s else 0
    ranked.append((avg, i, entry))

ranked.sort(reverse=True)
for rank, (avg, i, entry) in enumerate(ranked, 1):
    s = scores[i]
    detail = ", ".join(str(x) for x in s) if s else "N/A"
    print(f"\n{rank}位 (平均 {avg:.1f}/10, 各ラウンド: [{detail}])")
    print(entry)
