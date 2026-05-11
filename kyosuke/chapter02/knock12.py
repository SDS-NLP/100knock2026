path = "popular-names.txt"
with open(path, "r", encoding = "utf-8") as f:
    lines = f.readlines()
N = int(input("N:"))
for line in lines[-N:]:
    print(line, end = "")