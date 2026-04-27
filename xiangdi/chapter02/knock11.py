num = 10
count = 0

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    for line in f:
        if count >= num:
            break
        print(line, end="")
        count += 1

# head -n 10 /Users/caitlyn/Downloads/popular-names.txt