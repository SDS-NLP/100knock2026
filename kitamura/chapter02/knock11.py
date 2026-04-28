file = "popular-names.txt"
N = 10

with open(file, "r", encoding = "utf-8") as f:
    for i, line in enumerate(f): #iは０から
        if i>=N:
            break
        print(line)
        