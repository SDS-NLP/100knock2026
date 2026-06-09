from collections import defaultdict
sem_correct, sem_total = 0, 0
syn_correct, syn_total = 0, 0
per_category = defaultdict(lambda: [0, 0])

with open("q54_results.txt", encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        category = parts[0]
        gold = parts[4]
        pred = parts[5]
        
        is_correct = (pred == gold)
        per_category[category][1] += 1
        if is_correct:
            per_category[category][0] += 1
        if category.startswith("gram"):
            syn_total += 1
            if is_correct:
                syn_correct += 1
        else:
            sem_total += 1
            if is_correct:
                sem_correct += 1

print(f"意味的アナロジー: {sem_correct}/{sem_total} = {sem_correct/sem_total:.4f}")
print(f"文法的アナロジー: {syn_correct}/{syn_total} = {syn_correct/syn_total:.4f}")
print()
print("カテゴリ別:")
for cat, (c, t) in per_category.items():
    print(f"  {cat:30s} {c}/{t} = {c/t:.4f}")