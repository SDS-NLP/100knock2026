file = "popular-names.txt"
names = set()

with open(file, "r", encoding = "utf-8") as f:
    for line in f:
        cols = line.split()
        names.add(cols[0])

print(len(names))