path = "popular-names.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines[:10]:
    list = line.split("\t")
    print(list[0])