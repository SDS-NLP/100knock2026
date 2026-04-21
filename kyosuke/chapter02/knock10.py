path = "popular-names.txt"
with open(path, "r", encoding = "utf-8") as f:
    lines = f.readlines()
print(len(lines))