N = 10
filename = "data/popular-names.txt"

with open(filename, "r") as f:
    lines = f.readlines()
    result = "".join(lines[-N:])

print(result)

# tail -n 10 shiratori/chapter02/popular-names.txt
