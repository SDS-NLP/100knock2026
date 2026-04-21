path = "popular-names.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
unique_names = set()
for line in lines:
    name = line.split("\t")[0]
    unique_names.add(name)
print(len(unique_names))