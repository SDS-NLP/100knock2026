import random

with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

for line in lines:
    print(line, end="")

"unixコマンドはshuf popular-names.txt"
