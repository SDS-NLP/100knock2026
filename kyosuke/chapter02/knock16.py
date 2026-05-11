import random

path = "popular-names.txt"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()
random.shuffle(lines)