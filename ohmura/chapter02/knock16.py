import random

filename = "popular-names.txt"

with open(filename, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

for line in lines[:5]:
    print(line.strip())