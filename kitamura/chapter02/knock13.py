file = "popular-names.txt"
N = 10

with open(file,"r", encoding = "utf-8") as f:
    for i, line in enumerate(f):
        if i >= N:
            break

        replaced_line = line.replace("\t", " ")
        print(replaced_line)
