file = "popular-names.txt"

with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()


sorted_lines = sorted(
    lines, 
    key=lambda x: 1 / float(x.split("\t")[2])
)

for line in sorted_lines[:10]:
    print(line, end="")