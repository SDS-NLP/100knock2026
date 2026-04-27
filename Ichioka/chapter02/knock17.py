unique = set()

with open("popular-names.txt", encoding="utf-8") as f:
    for line in f:
        cols = line.split()
        if cols:
            unique.add(cols[0])

print(unique)