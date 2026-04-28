file = "popular-names.txt"
N = 10

with open(file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i>= N:
            break

        cols = line.split("\t")
        print(cols[0])