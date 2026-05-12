names = set()

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    for line in f:
        name = line.split()[0]
        names.add(name)

for name in names:
    print(name)

# cut -f 1 /Users/caitlyn/Downloads/popular-names.txt | 
# sort | uniq