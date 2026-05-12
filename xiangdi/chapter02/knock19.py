with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    lines = f.readlines()

sorted_lines = sorted(lines, key=lambda x: int(x.split()[2]), 
                      reverse=True)

for line in sorted_lines:
    print(line, end="")


# sort -k 3,3 -nr /Users/caitlyn/Downloads/popular-names.txt