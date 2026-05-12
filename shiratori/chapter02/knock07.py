filename = "shiratori/chapter02/popular-names.txt"

col1 = set()


with open(filename, "r") as f:
    for line in f:
        word = line.split()[0]
        col1.add(word)


for name in col1:
    print(name)

# cut -f1 shiratori/chapter02/popular-names.txt | sort | uniq
