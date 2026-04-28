N = 10

filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    for i in range(N):
        line = f.readline()
        cols = line.strip().split("\t")
        print(cols[0])

# head -n 10 shiratori/chapter02/popular-names.txt | cut -f1
