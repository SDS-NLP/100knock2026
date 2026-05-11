path = "popular-names.txt"
names = {}
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        name = line.split("\t")[0]
        if name in names:
            names[name] += 1
        else:
            names[name] = 1
sorted_names = sorted(names.items(), key = lambda x: x[1], reverse=True)
for name,count in sorted_names:
    print(name,count)