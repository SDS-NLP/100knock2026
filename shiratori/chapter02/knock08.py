filename = "shiratori/chapter02/popular-names.txt"


counts = {}

with open(filename, "r") as f:
    for line in f:
        cols = line.split()
        name = cols[0]

        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 1

items = list(counts.items())

sorted_items = sorted(items, key=lambda pair: pair[1], reverse=True)

for name, count in sorted_items:
    print(count, name)
