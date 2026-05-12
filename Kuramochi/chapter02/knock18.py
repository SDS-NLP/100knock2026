from collections import Counter

with open("popular-names.txt", encoding="utf-8") as f:
    first_cols = [line.split("\t")[0] for line in f if line.strip()]

for name, count in Counter(first_cols).most_common():
    print(count, name)

# 確認用のコマンド例:
# cut -f1 popular-names.txt | sort | uniq -c | sort -nr
