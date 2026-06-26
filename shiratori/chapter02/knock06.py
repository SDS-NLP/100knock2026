import random


N = 10

filename = "data/popular-names.txt"

with open(filename, "r") as f:
    lines = f.readlines()

random.shuffle(lines)
print("".join(lines))

# gshuf shiratori/chapter02/popular-names.txt
