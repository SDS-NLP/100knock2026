import random

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    lines = f.readlines()

random.shuffle(lines)

for line in lines:
    print(line, end="")

# shuf /Users/caitlyn/Downloads/popular-names.txt