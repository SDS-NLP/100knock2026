file = "popular-names.txt"

with open(file, "r", encoding="utf-8") as f:
    content = f.readlines()

for line in content[-N:]:
    print(line)
