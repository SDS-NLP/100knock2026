import random

with open("popular-names.txt", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

for line in lines:
    print(line, end="")