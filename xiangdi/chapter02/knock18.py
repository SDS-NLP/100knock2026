from collections import Counter

names = []

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    for line in f:
        names.append(line.split()[0])

counter = Counter(names)

for name, count in counter.most_common():
    print(name, count)

# cut -f 1 /Users/caitlyn/Downloads/popular-names.txt | sort | 
# uniq -c | sort -nr