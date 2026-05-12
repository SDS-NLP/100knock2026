from collections import Counter

counter = Counter()

with open("popular-names.txt", encoding="utf-8") as f:
    for line in f:
        cols = line.split()
        if cols:
            counter[cols[0]] += 1

for name, count in counter.most_common():
    print(name, count)