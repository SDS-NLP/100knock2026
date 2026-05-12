from collections import Counter

filename = "popular-names.txt"

with open(filename, "r", encoding="utf-8") as f:
    names = [line.split("\t")[0] for line in f]

name_counts = Counter(names)

for name, count in name_counts.most_common(10):
    print(f"{name}: {count}回")