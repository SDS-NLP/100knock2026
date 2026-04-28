count = 0
num = 10

with open("/Users/caitlyn/Downloads/popular-names.txt", 
          "r") as f:
    for line in f:
        if count >= num:
            break
        print(line.split()[0])
        count += 1

# head -n 10 /Users/caitlyn/Downloads/popular-names.txt | 
# cut -f 1