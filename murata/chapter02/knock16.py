import random

with open('popular-names.txt') as f:
    lines = f.readlines()

random.shuffle(lines)
for line in lines:
    print(line, end='')

# shuf popular-names.txt