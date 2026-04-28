count = 0
filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    for line in f:
        count += 1

print(count)

# wc -l popular-names.txt
