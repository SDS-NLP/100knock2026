N = 10
result = ""

filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    for i in range(N):
        line = f.readline()
        result += line


print(result)

# head -n 10 shiratori/chapter02/popular-names.txt
