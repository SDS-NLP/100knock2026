N = 10
head10 = ""

filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    for i in range(N):
        line = f.readline()
        head10 += line

result = head10.replace("\t", " ")

print(result)

# head -n 10 shiratori/chapter02/popular-names.txt | expand -t 1
