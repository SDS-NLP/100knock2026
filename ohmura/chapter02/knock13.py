filename = "popular-names.txt"
N = 10

with open("popular-names.txt", "r") as f:
    for i, line in enumerate(f):
        if i >= N:
            break

        clean_line = line.replace("\t", " ")

        print(clean_line.strip())