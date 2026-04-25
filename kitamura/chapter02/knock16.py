import random 
file = "popular-names.txt"

with open(file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

for line in lines:
    print(line)