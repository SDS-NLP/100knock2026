import random


N = 10

filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    lines = f.readlines()

random.shuffle(lines)
print("".join(lines))

# gshuf shiratori/chapter02/popular-names.txt
