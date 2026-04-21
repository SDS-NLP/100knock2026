path = "popular-names.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
sorted_names = sorted(lines, key=lambda x: int(x.split("\t")[2]), reverse=True)
for line in sorted_names[:10]:
    print(line,end="")