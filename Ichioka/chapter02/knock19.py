with open("popular-names.txt", encoding="utf-8") as f:
    lines = f.readlines()

lines.sort(key=lambda x: int(x.split()[2]), reverse=True)

for line in lines:
    print(line, end="")